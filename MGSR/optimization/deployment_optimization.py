import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import time
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

class DynamicComputationGraph(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 adaptive_config: Dict[str, Any] = None):
        super().__init__()
        
        self.model = model
        
        if adaptive_config is None:
            self.adaptive_config = {
                'max_sequence_length': 200,
                'dynamic_batching': True,
                'early_exit_threshold': 0.9,
                'layer_skipping': True,
                'adaptive_precision': True
            }
        else:
            self.adaptive_config = adaptive_config
        
        self.computation_stats = {}
        self.layer_execution_times = {}
        self.early_exit_points = []
        
        self._setup_early_exit_points()
        self._setup_layer_monitoring()
        
    def _setup_early_exit_points(self):
        layer_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                layer_count += 1
                if layer_count % 2 == 0:
                    exit_classifier = nn.Sequential(
                        nn.Linear(getattr(module, 'out_features', 512), 256),
                        nn.ReLU(),
                        nn.Linear(256, 1),
                        nn.Sigmoid()
                    )
                    self.early_exit_points.append((name, exit_classifier))
        
        self.early_exit_classifiers = nn.ModuleDict({
            name: classifier for name, classifier in self.early_exit_points
        })
    
    def _setup_layer_monitoring(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention, nn.LayerNorm)):
                self.layer_execution_times[name] = []
    
    def forward(self,
                inputs: torch.Tensor,
                dynamic_length: Optional[int] = None,
                confidence_threshold: float = None) -> Dict[str, torch.Tensor]:
        
        if confidence_threshold is None:
            confidence_threshold = self.adaptive_config['early_exit_threshold']
        
        batch_size = inputs.size(0)
        device = inputs.device
        
        if dynamic_length is not None:
            inputs = self._adjust_sequence_length(inputs, dynamic_length)
        
        intermediate_outputs = {}
        execution_path = []
        total_computation_time = 0.0
        
        x = inputs
        layer_idx = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention, nn.LayerNorm)):
                start_time = time.time()
                
                should_skip = self._should_skip_layer(name, x, layer_idx)
                
                if not should_skip:
                    x = module(x)
                    execution_path.append(name)
                
                end_time = time.time()
                layer_time = end_time - start_time
                total_computation_time += layer_time
                
                self.layer_execution_times[name].append(layer_time)
                if len(self.layer_execution_times[name]) > 100:
                    self.layer_execution_times[name].pop(0)
                
                if name in self.early_exit_classifiers:
                    confidence = self.early_exit_classifiers[name](x.mean(dim=1))
                    
                    if torch.mean(confidence) > confidence_threshold:
                        intermediate_outputs['early_exit'] = True
                        intermediate_outputs['exit_layer'] = name
                        intermediate_outputs['confidence'] = torch.mean(confidence).item()
                        break
                
                layer_idx += 1
        
        final_outputs = self._process_final_outputs(x)
        
        result = {
            'outputs': final_outputs,
            'intermediate_outputs': intermediate_outputs,
            'execution_path': execution_path,
            'computation_time': total_computation_time,
            'layers_executed': len(execution_path)
        }
        
        return result
    
    def _adjust_sequence_length(self, inputs: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = inputs.size(1)
        
        if current_length > target_length:
            return inputs[:, :target_length]
        elif current_length < target_length:
            padding = torch.zeros(
                inputs.size(0), target_length - current_length, inputs.size(2),
                device=inputs.device, dtype=inputs.dtype
            )
            return torch.cat([inputs, padding], dim=1)
        else:
            return inputs
    
    def _should_skip_layer(self, layer_name: str, x: torch.Tensor, layer_idx: int) -> bool:
        if not self.adaptive_config['layer_skipping']:
            return False
        
        if layer_idx < 2:
            return False
        
        if layer_name in self.layer_execution_times:
            avg_time = np.mean(self.layer_execution_times[layer_name][-10:])
            if avg_time > 0.01:
                input_variance = torch.var(x).item()
                if input_variance < 0.01:
                    return True
        
        return False
    
    def _process_final_outputs(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, 'output_projection'):
            return self.model.output_projection(x)
        else:
            return x
    
    def get_computation_statistics(self) -> Dict[str, Any]:
        stats = {
            'average_layer_times': {},
            'total_layers': len(self.layer_execution_times),
            'early_exit_rate': 0.0
        }
        
        for layer_name, times in self.layer_execution_times.items():
            if times:
                stats['average_layer_times'][layer_name] = np.mean(times)
        
        return stats

class BatchingOptimizer:
    def __init__(self,
                 max_batch_size: int = 32,
                 max_wait_time: float = 0.1,
                 padding_strategy: str = 'dynamic'):
        
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.padding_strategy = padding_strategy
        
        self.request_queue = queue.Queue()
        self.batch_processor = None
        self.is_running = False
        
    def start_batching(self, model: nn.Module, device: torch.device):
        self.is_running = True
        self.batch_processor = threading.Thread(
            target=self._batch_processing_loop,
            args=(model, device)
        )
        self.batch_processor.start()
    
    def stop_batching(self):
        self.is_running = False
        if self.batch_processor:
            self.batch_processor.join()
    
    def add_request(self, inputs: torch.Tensor, request_id: str) -> queue.Queue:
        result_queue = queue.Queue()
        self.request_queue.put((inputs, request_id, result_queue))
        return result_queue
    
    def _batch_processing_loop(self, model: nn.Module, device: torch.device):
        while self.is_running:
            batch_requests = []
            start_time = time.time()
            
            while (len(batch_requests) < self.max_batch_size and 
                   (time.time() - start_time) < self.max_wait_time):
                try:
                    request = self.request_queue.get(timeout=0.01)
                    batch_requests.append(request)
                except queue.Empty:
                    continue
            
            if batch_requests:
                self._process_batch(batch_requests, model, device)
    
    def _process_batch(self, batch_requests: List[Tuple], model: nn.Module, device: torch.device):
        inputs_list = [req[0] for req in batch_requests]
        request_ids = [req[1] for req in batch_requests]
        result_queues = [req[2] for req in batch_requests]
        
        batched_inputs = self._create_batch(inputs_list, device)
        
        with torch.no_grad():
            batch_outputs = model(batched_inputs)
        
        self._distribute_results(batch_outputs, result_queues, len(inputs_list))
    
    def _create_batch(self, inputs_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        if self.padding_strategy == 'dynamic':
            max_length = max(inp.size(1) for inp in inputs_list)
            
            padded_inputs = []
            for inp in inputs_list:
                if inp.size(1) < max_length:
                    padding = torch.zeros(
                        inp.size(0), max_length - inp.size(1), inp.size(2),
                        device=device, dtype=inp.dtype
                    )
                    padded_inp = torch.cat([inp, padding], dim=1)
                else:
                    padded_inp = inp
                
                padded_inputs.append(padded_inp)
            
            return torch.cat(padded_inputs, dim=0).to(device)
        
        else:
            return torch.cat(inputs_list, dim=0).to(device)
    
    def _distribute_results(self, batch_outputs: torch.Tensor, result_queues: List[queue.Queue], batch_size: int):
        if isinstance(batch_outputs, dict):
            for i, result_queue in enumerate(result_queues):
                individual_result = {}
                for key, value in batch_outputs.items():
                    if isinstance(value, torch.Tensor) and value.size(0) == batch_size:
                        individual_result[key] = value[i:i+1]
                    elif isinstance(value, list) and len(value) == batch_size:
                        individual_result[key] = [value[i]]
                    else:
                        individual_result[key] = value
                
                result_queue.put(individual_result)
        else:
            outputs_per_request = batch_outputs.size(0) // batch_size
            for i, result_queue in enumerate(result_queues):
                start_idx = i * outputs_per_request
                end_idx = (i + 1) * outputs_per_request
                result_queue.put(batch_outputs[start_idx:end_idx])

class ModelCaching:
    def __init__(self,
                 cache_size: int = 1000,
                 ttl_seconds: int = 3600):
        
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.cache:
            current_time = time.time()
            
            if current_time - self.creation_times[key] > self.ttl_seconds:
                self._remove_key(key)
                return None
            
            self.access_times[key] = current_time
            return self.cache[key]
        
        return None
    
    def put(self, key: str, value: torch.Tensor):
        current_time = time.time()
        
        if len(self.cache) >= self.cache_size:
            self._evict_lru()
        
        self.cache[key] = value.detach().clone()
        self.access_times[key] = current_time
        self.creation_times[key] = current_time
    
    def _evict_lru(self):
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def _remove_key(self, key: str):
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.creation_times[key]
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_utilization': len(self.cache) / self.cache_size
        }

class AsyncInferenceEngine:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 max_workers: int = 4):
        
        self.model = model
        self.device = device
        self.max_workers = max_workers
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model_cache = ModelCaching()
        self.batching_optimizer = BatchingOptimizer()
        
        self.request_count = 0
        self.total_inference_time = 0.0
        
    async def async_inference(self,
                             inputs: torch.Tensor,
                             request_id: Optional[str] = None) -> Dict[str, Any]:
        
        if request_id is None:
            request_id = f"req_{self.request_count}"
            self.request_count += 1
        
        cache_key = self._generate_cache_key(inputs)
        cached_result = self.model_cache.get(cache_key)
        
        if cached_result is not None:
            return {
                'outputs': cached_result,
                'from_cache': True,
                'request_id': request_id
            }
        
        loop = asyncio.get_event_loop()
        
        start_time = time.time()
        result = await loop.run_in_executor(
            self.executor,
            self._sync_inference,
            inputs,
            request_id
        )
        end_time = time.time()
        
        inference_time = end_time - start_time
        self.total_inference_time += inference_time
        
        self.model_cache.put(cache_key, result['outputs'])
        
        result.update({
            'inference_time': inference_time,
            'from_cache': False,
            'request_id': request_id
        })
        
        return result
    
    def _sync_inference(self, inputs: torch.Tensor, request_id: str) -> Dict[str, Any]:
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return {'outputs': outputs}
    
    def _generate_cache_key(self, inputs: torch.Tensor) -> str:
        input_hash = hash(inputs.cpu().numpy().tobytes())
        return f"input_{input_hash}_{inputs.shape}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        avg_inference_time = (self.total_inference_time / self.request_count 
                             if self.request_count > 0 else 0.0)
        
        return {
            'total_requests': self.request_count,
            'average_inference_time': avg_inference_time,
            'total_inference_time': self.total_inference_time,
            'cache_stats': self.model_cache.get_cache_stats()
        }

class ModelOptimizer:
    def __init__(self):
        self.optimization_history = []
    
    def optimize_for_deployment(self,
                               model: nn.Module,
                               sample_inputs: torch.Tensor,
                               optimization_config: Dict[str, Any] = None) -> nn.Module:
        
        if optimization_config is None:
            optimization_config = {
                'enable_jit': True,
                'enable_quantization': True,
                'enable_pruning': False,
                'enable_fusion': True
            }
        
        optimized_model = model
        optimization_steps = []
        
        if optimization_config.get('enable_fusion', False):
            candidate = self._apply_operator_fusion(optimized_model)
            if candidate is not optimized_model:
                optimization_steps.append('operator_fusion')
            optimized_model = candidate
        
        if optimization_config.get('enable_jit', False):
            candidate = self._apply_jit_compilation(optimized_model, sample_inputs)
            if candidate is not optimized_model:
                optimization_steps.append('jit_compilation')
            optimized_model = candidate
        
        if optimization_config.get('enable_quantization', False):
            candidate = self._apply_quantization(optimized_model, sample_inputs)
            if candidate is not optimized_model:
                optimization_steps.append('quantization')
            optimized_model = candidate
        
        if optimization_config.get('enable_pruning', False):
            optimized_model = self._apply_pruning(optimized_model)
            optimization_steps.append('pruning')
        
        self.optimization_history.append({
            'steps': optimization_steps,
            'config': optimization_config
        })
        
        return optimized_model
    
    def _apply_operator_fusion(self, model: nn.Module) -> nn.Module:
        # Linear/ReLU reconstruction is not operator fusion and must not be
        # reported as an optimization. Keep the hook for architecture-specific
        # fusion implementations.
        return model
    
    def _fuse_sequential_layers(self, sequential: nn.Sequential) -> Optional[nn.Module]:
        layers = list(sequential.children())
        
        if len(layers) >= 2:
            if (isinstance(layers[0], nn.Linear) and 
                isinstance(layers[1], (nn.ReLU, nn.GELU))):
                
                fused = nn.Sequential(
                    layers[0],
                    layers[1]
                )
                
                if len(layers) > 2:
                    for layer in layers[2:]:
                        fused.add_module(str(len(fused)), layer)
                
                return fused
        
        return None
    
    def _apply_jit_compilation(self, model: nn.Module, sample_inputs: torch.Tensor) -> nn.Module:
        model.eval()
        
        try:
            traced_model = torch.jit.trace(model, sample_inputs)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            return traced_model
        except Exception as e:
            print(f"JIT compilation failed: {e}")
            return model
    
    def _apply_quantization(self, model: nn.Module, sample_inputs: torch.Tensor) -> nn.Module:
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            print(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        try:
            import torch.nn.utils.prune as prune
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
            
            return model
        except Exception as e:
            print(f"Pruning failed: {e}")
            return model

class DeploymentBenchmark:
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_deployment_optimizations(self,
                                         original_model: nn.Module,
                                         optimized_model: nn.Module,
                                         test_inputs,
                                         device: torch.device,
                                         num_runs: int = 100) -> Dict[str, Any]:
        
        original_stats = self._benchmark_model(original_model, test_inputs, device, num_runs)
        optimized_stats = self._benchmark_model(optimized_model, test_inputs, device, num_runs)
        
        comparison = {
            'original_inference_time_ms': original_stats['avg_inference_time'] * 1000,
            'optimized_inference_time_ms': optimized_stats['avg_inference_time'] * 1000,
            'speedup': original_stats['avg_inference_time'] / optimized_stats['avg_inference_time'],
            'original_memory_mb': original_stats['memory_usage'] / (1024 * 1024),
            'optimized_memory_mb': optimized_stats['memory_usage'] / (1024 * 1024),
            'memory_reduction': 1 - (optimized_stats['memory_usage'] / original_stats['memory_usage']),
            'original_model_size_mb': self._get_model_size(original_model) / (1024 * 1024),
            'optimized_model_size_mb': self._get_model_size(optimized_model) / (1024 * 1024),
            'size_reduction': 1 - (self._get_model_size(optimized_model) / self._get_model_size(original_model))
        }
        
        self.benchmark_results = comparison
        return comparison
    
    def _benchmark_model(self,
                        model: nn.Module,
                        test_inputs,
                        device: torch.device,
                        num_runs: int) -> Dict[str, float]:
        
        model.eval()
        model = model.to(device)
        test_inputs = self._move_to_device(test_inputs, device)
        
        inference_times = []
        memory_usage = 0
        
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
                start_time = time.time()
                _ = model(**test_inputs, mode='inference') if isinstance(test_inputs, dict) else model(test_inputs)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                
                if device.type == 'cuda':
                    memory_usage = max(memory_usage, torch.cuda.max_memory_allocated())
        
        return {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'memory_usage': memory_usage
        }
    
    def _get_model_size(self, model: nn.Module) -> int:
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size

    def _move_to_device(self, value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: self._move_to_device(item, device) for key, item in value.items()}
        return value
