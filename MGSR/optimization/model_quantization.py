import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import math

class QuantizationAwareTraining(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 quantization_config: Dict[str, any] = None):
        super().__init__()
        
        self.model = model
        
        self.quantization_config = {
            'weight_bits': 8,
            'activation_bits': 8,
            'quantization_scheme': 'symmetric',
            'per_channel': True,
            'observer_type': 'minmax'
        }
        if quantization_config:
            self.quantization_config.update(quantization_config)
        
        self.quantized_layers = nn.ModuleList()
        self.quantized_layer_names = []
        self._prepare_quantization()
        
        self.calibration_data = []
        self.is_calibrated = False
        
    def _prepare_quantization(self):
        for name, module in self.model.named_modules():
            # MultiheadAttention reads out_proj.weight directly instead of
            # calling the module, so wrapping that projection breaks attention.
            if isinstance(module, nn.Linear) and not name.endswith('out_proj'):
                quantized_module = QuantizedLinear(
                    module,
                    weight_bits=self.quantization_config['weight_bits'],
                    activation_bits=self.quantization_config['activation_bits'],
                    quantization_scheme=self.quantization_config['quantization_scheme'],
                    per_channel=self.quantization_config['per_channel']
                )
                self.quantized_layers.append(quantized_module)
                self.quantized_layer_names.append(name)
    
    def forward(self, *args, **kwargs):
        if self.training:
            return self._forward_training(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)
    
    def _forward_training(self, *args, **kwargs):
        original_modules = {}
        
        for name, quantized_module in zip(self.quantized_layer_names, self.quantized_layers):
            original_module = self._get_module_by_name(self.model, name)
            original_modules[name] = original_module
            self._set_module_by_name(self.model, name, quantized_module)
        
        try:
            outputs = self.model(*args, **kwargs)
            for quantized_module in self.quantized_layers:
                quantized_module.is_calibrated = True
        finally:
            for name, original_module in original_modules.items():
                self._set_module_by_name(self.model, name, original_module)
        
        return outputs
    
    def _forward_inference(self, *args, **kwargs):
        if not self.is_calibrated:
            self._calibrate()
        
        return self._forward_training(*args, **kwargs)
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> nn.Module:
        parts = name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        return module
    
    def _set_module_by_name(self, model: nn.Module, name: str, new_module: nn.Module):
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def add_calibration_data(self, data):
        raise NotImplementedError(
            "MGSR calibration must run representative batches through each layer; "
            "raw batch tensors cannot be reused as every layer's activation."
        )
    
    def _calibrate(self):
        if not self.calibration_data:
            return
        
        self.is_calibrated = all(module.is_calibrated for module in self.quantized_layers)

class QuantizedLinear(nn.Module):
    def __init__(self,
                 original_layer: nn.Module,
                 weight_bits: int = 8,
                 activation_bits: int = 8,
                 quantization_scheme: str = 'symmetric',
                 per_channel: bool = True):
        super().__init__()
        
        self.original_layer = original_layer
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quantization_scheme = quantization_scheme
        self.per_channel = per_channel
        
        self.weight_quantizer = WeightQuantizer(
            bits=weight_bits,
            scheme=quantization_scheme,
            per_channel=per_channel
        )
        
        self.activation_quantizer = ActivationQuantizer(
            bits=activation_bits,
            scheme=quantization_scheme
        )
        
        self.is_calibrated = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            quantized_weight = self.weight_quantizer(self.original_layer.weight)
            quantized_input = self.activation_quantizer(x)
            
            output = F.linear(quantized_input, quantized_weight, self.original_layer.bias)
            
            return output
        else:
            if not self.is_calibrated:
                return self.original_layer(x)
            
            quantized_weight = self.weight_quantizer(self.original_layer.weight)
            quantized_input = self.activation_quantizer(x)
            
            output = F.linear(quantized_input, quantized_weight, self.original_layer.bias)
            
            return output
    
    def calibrate(self, calibration_data: List[torch.Tensor]):
        self.activation_quantizer.calibrate(calibration_data)
        self.weight_quantizer.calibrate([self.original_layer.weight])
        self.is_calibrated = True

class WeightQuantizer(nn.Module):
    def __init__(self,
                 bits: int = 8,
                 scheme: str = 'symmetric',
                 per_channel: bool = True):
        super().__init__()
        
        self.bits = bits
        self.scheme = scheme
        self.per_channel = per_channel
        
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        
        self.scale = None
        self.zero_point = None
        
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.scale is None or self.zero_point is None:
            self._compute_quantization_params(weight)
        
        return self._quantize_dequantize(weight)
    
    def _compute_quantization_params(self, weight: torch.Tensor):
        if self.per_channel:
            dim = 0
            weight_view = weight.view(weight.size(0), -1)
            min_vals = torch.min(weight_view, dim=1)[0]
            max_vals = torch.max(weight_view, dim=1)[0]
        else:
            min_vals = torch.min(weight)
            max_vals = torch.max(weight)
        
        if self.scheme == 'symmetric':
            abs_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))
            self.scale = abs_max / (2 ** (self.bits - 1) - 1)
            self.zero_point = torch.zeros_like(self.scale, dtype=torch.int32)
        else:
            self.scale = (max_vals - min_vals) / (2 ** self.bits - 1)
            self.zero_point = torch.round(-min_vals / self.scale).to(torch.int32)
        
        self.scale = torch.clamp(self.scale, min=1e-8)
    
    def _quantize_dequantize(self, weight: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            scale = self.scale.view(-1, *([1] * (weight.dim() - 1)))
            zero_point = self.zero_point.view(-1, *([1] * (weight.dim() - 1)))
        else:
            scale = self.scale
            zero_point = self.zero_point
        
        quantized = torch.round(weight / scale + zero_point)
        quantized = torch.clamp(quantized, self.qmin, self.qmax)
        
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def calibrate(self, calibration_data: List[torch.Tensor]):
        if calibration_data:
            weight = calibration_data[0]
            self._compute_quantization_params(weight)

class ActivationQuantizer(nn.Module):
    def __init__(self,
                 bits: int = 8,
                 scheme: str = 'symmetric',
                 observer_type: str = 'minmax'):
        super().__init__()
        
        self.bits = bits
        self.scheme = scheme
        self.observer_type = observer_type
        
        self.qmin = 0 if scheme == 'asymmetric' else -(2 ** (bits - 1))
        self.qmax = 2 ** bits - 1 if scheme == 'asymmetric' else 2 ** (bits - 1) - 1
        
        self.scale = None
        self.zero_point = None
        
        self.min_val = None
        self.max_val = None
        
        if observer_type == 'ema':
            self.momentum = 0.1
        elif observer_type == 'percentile':
            self.percentile = 99.9
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_statistics(x)
        
        if self.scale is None or self.zero_point is None:
            return x
        
        return self._quantize_dequantize(x)
    
    def _update_statistics(self, x: torch.Tensor):
        current_min = torch.min(x)
        current_max = torch.max(x)
        
        if self.observer_type == 'minmax':
            if self.min_val is None:
                self.min_val = current_min
                self.max_val = current_max
            else:
                self.min_val = torch.min(self.min_val, current_min)
                self.max_val = torch.max(self.max_val, current_max)
        
        elif self.observer_type == 'ema':
            if self.min_val is None:
                self.min_val = current_min
                self.max_val = current_max
            else:
                self.min_val = (1 - self.momentum) * self.min_val + self.momentum * current_min
                self.max_val = (1 - self.momentum) * self.max_val + self.momentum * current_max
        
        elif self.observer_type == 'percentile':
            x_flat = x.view(-1)
            self.min_val = torch.quantile(x_flat, (100 - self.percentile) / 100)
            self.max_val = torch.quantile(x_flat, self.percentile / 100)
        
        self._compute_quantization_params()
    
    def _compute_quantization_params(self):
        if self.min_val is None or self.max_val is None:
            return
        
        if self.scheme == 'symmetric':
            abs_max = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
            self.scale = abs_max / (2 ** (self.bits - 1) - 1)
            self.zero_point = torch.tensor(0, dtype=torch.int32)
        else:
            self.scale = (self.max_val - self.min_val) / (2 ** self.bits - 1)
            self.zero_point = torch.round(-self.min_val / self.scale).to(torch.int32)
        
        self.scale = torch.clamp(self.scale, min=1e-8)
    
    def _quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        quantized = torch.round(x / self.scale + self.zero_point)
        quantized = torch.clamp(quantized, self.qmin, self.qmax)
        
        dequantized = (quantized - self.zero_point) * self.scale
        
        return dequantized
    
    def calibrate(self, calibration_data: List[torch.Tensor]):
        for data in calibration_data:
            self._update_statistics(data)

class PostTrainingQuantization:
    def __init__(self,
                 model: nn.Module,
                 calibration_dataset: torch.utils.data.DataLoader,
                 quantization_config: Dict[str, any] = None):
        
        self.model = model
        self.calibration_dataset = calibration_dataset
        
        if quantization_config is None:
            self.quantization_config = {
                'weight_bits': 8,
                'activation_bits': 8,
                'quantization_scheme': 'symmetric',
                'calibration_samples': 100
            }
        else:
            self.quantization_config = quantization_config
        
        self.quantized_model = None
    
    def quantize(self) -> nn.Module:
        self.model.eval()
        
        quantized_model = self._create_quantized_model()
        
        self._calibrate_model(quantized_model)
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _create_quantized_model(self) -> nn.Module:
        quantized_model = QuantizationAwareTraining(
            self.model, self.quantization_config
        )
        return quantized_model
    
    def _calibrate_model(self, quantized_model: QuantizationAwareTraining):
        sample_count = 0
        max_samples = self.quantization_config['calibration_samples']
        
        quantized_model.model.eval()
        for module in quantized_model.quantized_layers:
            module.train()
        with torch.no_grad():
            for batch_data in self.calibration_dataset:
                if sample_count >= max_samples:
                    break
                
                if isinstance(batch_data, (list, tuple)):
                    quantized_model(batch_data[0])
                elif isinstance(batch_data, dict):
                    quantized_model(**batch_data)
                else:
                    quantized_model(batch_data)
                
                sample_count += 1
        
        quantized_model.is_calibrated = True
        quantized_model.eval()

class DynamicQuantization(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 target_modules: List[str] = None):
        super().__init__()
        
        self.model = model
        
        if target_modules is None:
            self.target_modules = ['Linear', 'Conv1d', 'Conv2d']
        else:
            self.target_modules = target_modules
        
        self._apply_dynamic_quantization()
    
    def _apply_dynamic_quantization(self):
        for name, module in self.model.named_modules():
            if any(target in str(type(module)) for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    quantized_module = DynamicQuantizedLinear(module)
                    self._replace_module(name, quantized_module)
    
    def _replace_module(self, name: str, new_module: nn.Module):
        parts = name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class DynamicQuantizedLinear(nn.Module):
    def __init__(self, original_layer: nn.Linear):
        super().__init__()
        
        self.original_layer = original_layer
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scale, x_zero_point = self._compute_activation_quantization_params(x)
        w_scale, w_zero_point = self._compute_weight_quantization_params(self.weight)
        
        x_quantized = torch.quantize_per_tensor(x, x_scale, x_zero_point, torch.quint8)
        w_quantized = torch.quantize_per_tensor(self.weight, w_scale, w_zero_point, torch.qint8)
        
        # This module performs dynamic quantize-dequantize simulation so it
        # remains portable across CPU backends without packed-weight kernels.
        return F.linear(x_quantized.dequantize(), w_quantized.dequantize(), self.bias)
    
    def _compute_activation_quantization_params(self, x: torch.Tensor) -> Tuple[float, int]:
        x_min = torch.min(x)
        x_max = torch.max(x)
        
        scale = torch.clamp((x_max - x_min) / 255.0, min=1e-8)
        zero_point = int(torch.round(-x_min / scale))
        zero_point = max(0, min(255, zero_point))
        
        return scale.item(), zero_point
    
    def _compute_weight_quantization_params(self, weight: torch.Tensor) -> Tuple[float, int]:
        w_min = torch.min(weight)
        w_max = torch.max(weight)
        
        scale = torch.clamp(torch.maximum(abs(w_min), abs(w_max)) / 127.0, min=1e-8)
        zero_point = 0
        
        return scale.item(), zero_point

class MixedPrecisionTraining(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 precision_config: Dict[str, str] = None):
        super().__init__()
        
        self.model = model
        
        if precision_config is None:
            self.precision_config = {
                'attention': 'fp16',
                'feedforward': 'fp16',
                'embedding': 'fp32',
                'output': 'fp32'
            }
        else:
            self.precision_config = precision_config
        
        self._apply_mixed_precision()
        
        self.scaler = torch.cuda.amp.GradScaler()
    
    def _apply_mixed_precision(self):
        for name, module in self.model.named_modules():
            precision = self._get_module_precision(name, module)
            
            if precision == 'fp16':
                module.half()
            elif precision == 'fp32':
                module.float()
    
    def _get_module_precision(self, name: str, module: nn.Module) -> str:
        for key, precision in self.precision_config.items():
            if key.lower() in name.lower() or key.lower() in str(type(module)).lower():
                return precision
        
        return 'fp16'
    
    def forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast():
            return self.model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor):
        self.scaler.scale(loss).backward()
    
    def step(self, optimizer: torch.optim.Optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()

class QuantizationBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_model(self,
                       original_model: nn.Module,
                       quantized_model: nn.Module,
                       test_data: torch.utils.data.DataLoader,
                       device: torch.device) -> Dict[str, any]:
        
        original_results = self._evaluate_model(original_model, test_data, device)
        quantized_results = self._evaluate_model(quantized_model, test_data, device)
        
        model_size_original = self._get_model_size(original_model)
        model_size_quantized = self._get_model_size(quantized_model)
        
        inference_time_original = self._measure_inference_time(original_model, test_data, device)
        inference_time_quantized = self._measure_inference_time(quantized_model, test_data, device)
        
        results = {
            'accuracy_original': original_results['accuracy'],
            'accuracy_quantized': quantized_results['accuracy'],
            'accuracy_drop': original_results['accuracy'] - quantized_results['accuracy'],
            'model_size_original_mb': model_size_original / (1024 * 1024),
            'model_size_quantized_mb': model_size_quantized / (1024 * 1024),
            'compression_ratio': model_size_original / model_size_quantized,
            'inference_time_original_ms': inference_time_original * 1000,
            'inference_time_quantized_ms': inference_time_quantized * 1000,
            'speedup': inference_time_original / inference_time_quantized
        }
        
        self.results = results
        return results
    
    def _evaluate_model(self, model: nn.Module, test_data: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in test_data:
                if isinstance(batch_data, dict):
                    inputs = self._move_to_device(batch_data, device)
                    targets = inputs.get('target_items')
                    if targets is None:
                        raise ValueError("MGSR benchmark batches must include target_items")
                    outputs = model(**inputs, mode='inference')
                elif isinstance(batch_data, (list, tuple)):
                    inputs, targets = batch_data
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                else:
                    raise ValueError("Quantization benchmark data must provide (inputs, targets)")
                
                if isinstance(outputs, dict):
                    if 'final_rankings' in outputs:
                        predicted = torch.tensor(
                            [ranking[0] if ranking else -1 for ranking in outputs['final_rankings']],
                            device=device
                        )
                    else:
                        raise ValueError("Model dictionary output must include final_rankings")
                else:
                    predicted = torch.argmax(outputs, dim=1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy}
    
    def _get_model_size(self, model: nn.Module) -> int:
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def _measure_inference_time(self, model: nn.Module, test_data: torch.utils.data.DataLoader, device: torch.device) -> float:
        model.eval()
        
        total_time = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in test_data:
                if num_batches >= 10:
                    break
                
                if isinstance(batch_data, dict):
                    inputs = self._move_to_device(batch_data, device)
                    call = lambda: model(**inputs, mode='inference')
                elif isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0]
                    inputs = inputs.to(device)
                    call = lambda: model(inputs)
                else:
                    inputs = batch_data
                    inputs = inputs.to(device)
                    call = lambda: model(inputs)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                
                if device.type == 'cuda':
                    start_time.record()
                else:
                    import time
                    start_time = time.time()
                
                _ = call()
                
                if device.type == 'cuda':
                    end_time.record()
                    torch.cuda.synchronize()
                    batch_time = start_time.elapsed_time(end_time) / 1000.0
                else:
                    batch_time = time.time() - start_time
                
                total_time += batch_time
                num_batches += 1
        
        return total_time / num_batches if num_batches > 0 else 0.0

    def _move_to_device(self, value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: self._move_to_device(item, device) for key, item in value.items()}
        return value
