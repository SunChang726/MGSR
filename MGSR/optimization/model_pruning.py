import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import math


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _prepare_batch(batch_data, device):
    if isinstance(batch_data, dict):
        inputs = _move_to_device(batch_data, device)
        return inputs, inputs.get('target_items')
    if isinstance(batch_data, (list, tuple)):
        inputs = _move_to_device(batch_data[0], device)
        targets = _move_to_device(batch_data[1], device) if len(batch_data) > 1 else None
        return inputs, targets
    return _move_to_device(batch_data, device), None


def _forward_model(model, inputs):
    return model(**inputs) if isinstance(inputs, dict) else model(inputs)


class StructuredPruning(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 pruning_config: Dict[str, any] = None):
        super().__init__()
        
        self.model = model
        
        self.pruning_config = {
            'pruning_ratio': 0.5,
            'pruning_method': 'magnitude',
            'structured_type': 'channel',
            'granularity': 'layer'
        }
        if pruning_config:
            self.pruning_config.update(pruning_config)
        
        self.pruning_masks = {}
        self.importance_scores = {}
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def compute_importance_scores(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.model.eval()
        
        if self.pruning_config['pruning_method'] == 'magnitude':
            self._compute_magnitude_scores()
        elif self.pruning_config['pruning_method'] == 'gradient':
            self._compute_gradient_scores(dataloader, device)
        elif self.pruning_config['pruning_method'] == 'fisher':
            self._compute_fisher_scores(dataloader, device)
        elif self.pruning_config['pruning_method'] == 'taylor':
            self._compute_taylor_scores(dataloader, device)
    
    def _compute_magnitude_scores(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight.data
                
                if self.pruning_config['structured_type'] == 'channel':
                    if len(weight.shape) >= 2:
                        scores = torch.norm(weight, dim=tuple(range(1, len(weight.shape))))
                    else:
                        scores = torch.abs(weight)
                elif self.pruning_config['structured_type'] == 'filter':
                    if len(weight.shape) >= 2:
                        scores = torch.norm(weight, dim=tuple(range(1, len(weight.shape))))
                    else:
                        scores = torch.abs(weight)
                else:
                    scores = torch.abs(weight.view(-1))
                
                self.importance_scores[name] = scores
    
    def _compute_gradient_scores(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.model.train()
        
        gradient_accumulator = {}
        
        for batch_data in dataloader:
            self.model.zero_grad()
            
            inputs, targets = _prepare_batch(batch_data, device)
            outputs = _forward_model(self.model, inputs)
            
            if isinstance(outputs, dict) and 'total_loss' in outputs:
                loss = outputs['total_loss']
            elif targets is not None:
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = torch.mean(outputs)
            
            loss.backward()
            
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and module.weight.grad is not None:
                    grad = module.weight.grad.data
                    
                    if name not in gradient_accumulator:
                        gradient_accumulator[name] = torch.zeros_like(grad)
                    
                    gradient_accumulator[name] += torch.abs(grad)
        
        for name, accumulated_grad in gradient_accumulator.items():
            if self.pruning_config['structured_type'] == 'channel':
                scores = torch.norm(accumulated_grad, dim=tuple(range(1, len(accumulated_grad.shape))))
            elif self.pruning_config['structured_type'] == 'filter':
                scores = torch.norm(accumulated_grad, dim=tuple(range(1, len(accumulated_grad.shape))))
            else:
                scores = accumulated_grad.view(-1)
            
            self.importance_scores[name] = scores
    
    def _compute_fisher_scores(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.model.train()
        
        fisher_accumulator = {}
        
        for batch_data in dataloader:
            self.model.zero_grad()
            
            inputs, _ = _prepare_batch(batch_data, device)
            outputs = _forward_model(self.model, inputs)
            
            if isinstance(outputs, dict) and 'total_loss' in outputs:
                loss = outputs['total_loss']
            else:
                loss = torch.mean(outputs)
            
            loss.backward()
            
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and module.weight.grad is not None:
                    grad = module.weight.grad.data
                    
                    if name not in fisher_accumulator:
                        fisher_accumulator[name] = torch.zeros_like(grad)
                    
                    fisher_accumulator[name] += grad ** 2
        
        for name, fisher_info in fisher_accumulator.items():
            if self.pruning_config['structured_type'] == 'channel':
                scores = torch.norm(fisher_info, dim=tuple(range(1, len(fisher_info.shape))))
            elif self.pruning_config['structured_type'] == 'filter':
                scores = torch.norm(fisher_info, dim=tuple(range(1, len(fisher_info.shape))))
            else:
                scores = fisher_info.view(-1)
            
            self.importance_scores[name] = scores
    
    def _compute_taylor_scores(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.model.train()
        
        taylor_accumulator = {}
        
        for batch_data in dataloader:
            self.model.zero_grad()
            
            inputs, _ = _prepare_batch(batch_data, device)
            outputs = _forward_model(self.model, inputs)
            
            if isinstance(outputs, dict) and 'total_loss' in outputs:
                loss = outputs['total_loss']
            else:
                loss = torch.mean(outputs)
            
            loss.backward()
            
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and module.weight.grad is not None:
                    weight = module.weight.data
                    grad = module.weight.grad.data
                    
                    taylor_score = torch.abs(weight * grad)
                    
                    if name not in taylor_accumulator:
                        taylor_accumulator[name] = torch.zeros_like(taylor_score)
                    
                    taylor_accumulator[name] += taylor_score
        
        for name, taylor_scores in taylor_accumulator.items():
            if self.pruning_config['structured_type'] == 'channel':
                scores = torch.norm(taylor_scores, dim=tuple(range(1, len(taylor_scores.shape))))
            elif self.pruning_config['structured_type'] == 'filter':
                scores = torch.norm(taylor_scores, dim=tuple(range(1, len(taylor_scores.shape))))
            else:
                scores = taylor_scores.view(-1)
            
            self.importance_scores[name] = scores
    
    def generate_pruning_masks(self):
        if self.pruning_config['granularity'] == 'global':
            self._generate_global_masks()
        else:
            self._generate_layer_wise_masks()
    
    def _generate_global_masks(self):
        all_scores = []
        score_to_layer = []
        
        for name, scores in self.importance_scores.items():
            all_scores.extend(scores.cpu().numpy().tolist())
            score_to_layer.extend([(name, i) for i in range(len(scores))])
        
        all_scores = np.array(all_scores)
        sorted_indices = np.argsort(all_scores)
        
        num_to_prune = int(len(all_scores) * self.pruning_config['pruning_ratio'])
        indices_to_prune = sorted_indices[:num_to_prune]
        
        for name in self.importance_scores.keys():
            self.pruning_masks[name] = torch.ones_like(self.importance_scores[name], dtype=torch.bool)
        
        for idx in indices_to_prune:
            layer_name, element_idx = score_to_layer[idx]
            self.pruning_masks[layer_name][element_idx] = False
    
    def _generate_layer_wise_masks(self):
        for name, scores in self.importance_scores.items():
            num_elements = len(scores)
            num_to_prune = int(num_elements * self.pruning_config['pruning_ratio'])
            
            _, indices_to_prune = torch.topk(scores, num_to_prune, largest=False)
            
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[indices_to_prune] = False
            
            self.pruning_masks[name] = mask
    
    def apply_pruning(self):
        for name, module in self.model.named_modules():
            if name in self.pruning_masks:
                mask = self.pruning_masks[name]
                
                if isinstance(module, nn.Linear):
                    self._prune_linear_layer(module, mask)
                elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    self._prune_conv_layer(module, mask)
    
    def _prune_linear_layer(self, layer: nn.Linear, mask: torch.Tensor):
        if self.pruning_config['structured_type'] == 'channel':
            layer.weight.data *= mask.to(layer.weight.dtype).unsqueeze(1)
            if layer.bias is not None:
                layer.bias.data *= mask.to(layer.bias.dtype)
        else:
            layer.weight.data *= mask.float().view_as(layer.weight.data)
    
    def _prune_conv_layer(self, layer: Union[nn.Conv1d, nn.Conv2d], mask: torch.Tensor):
        if self.pruning_config['structured_type'] == 'channel':
            shape = [mask.numel()] + [1] * (layer.weight.dim() - 1)
            layer.weight.data *= mask.to(layer.weight.dtype).view(shape)
            if layer.bias is not None:
                layer.bias.data *= mask.to(layer.bias.dtype)
        else:
            layer.weight.data *= mask.float().view_as(layer.weight.data)

class UnstructuredPruning(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 pruning_config: Dict[str, any] = None):
        super().__init__()
        
        self.model = model
        
        if pruning_config is None:
            self.pruning_config = {
                'pruning_ratio': 0.5,
                'pruning_method': 'magnitude',
                'sparsity_type': 'unstructured'
            }
        else:
            self.pruning_config = pruning_config
        
        self.pruning_masks = {}
        
    def forward(self, *args, **kwargs):
        self._apply_masks()
        return self.model(*args, **kwargs)
    
    def _apply_masks(self):
        for name, module in self.model.named_modules():
            if name in self.pruning_masks:
                mask = self.pruning_masks[name]
                module.weight.data *= mask
    
    def compute_pruning_masks(self):
        if self.pruning_config['pruning_method'] == 'magnitude':
            self._compute_magnitude_masks()
        elif self.pruning_config['pruning_method'] == 'random':
            self._compute_random_masks()
        elif self.pruning_config['pruning_method'] == 'snip':
            self._compute_snip_masks()
    
    def _compute_magnitude_masks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight.data
                weight_magnitude = torch.abs(weight)
                
                num_elements = weight.numel()
                num_to_prune = int(num_elements * self.pruning_config['pruning_ratio'])
                
                threshold = torch.topk(weight_magnitude.view(-1), num_to_prune, largest=False)[0][-1]
                
                mask = (weight_magnitude > threshold).float()
                self.pruning_masks[name] = mask
    
    def _compute_random_masks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight.data
                
                num_elements = weight.numel()
                num_to_keep = int(num_elements * (1 - self.pruning_config['pruning_ratio']))
                
                mask = torch.zeros_like(weight)
                flat_mask = mask.view(-1)
                
                indices_to_keep = torch.randperm(num_elements)[:num_to_keep]
                flat_mask[indices_to_keep] = 1.0
                
                self.pruning_masks[name] = mask
    
    def _compute_snip_masks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight.data
                
                if hasattr(module.weight, 'grad') and module.weight.grad is not None:
                    grad = module.weight.grad.data
                    snip_score = torch.abs(weight * grad)
                else:
                    snip_score = torch.abs(weight)
                
                num_elements = weight.numel()
                num_to_prune = int(num_elements * self.pruning_config['pruning_ratio'])
                
                threshold = torch.topk(snip_score.view(-1), num_to_prune, largest=False)[0][-1]
                
                mask = (snip_score > threshold).float()
                self.pruning_masks[name] = mask

class GradualPruning:
    def __init__(self,
                 model: nn.Module,
                 initial_sparsity: float = 0.0,
                 final_sparsity: float = 0.9,
                 pruning_frequency: int = 100,
                 pruning_schedule: str = 'polynomial'):
        
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.pruning_schedule = pruning_schedule
        
        self.current_step = 0
        self.pruning_masks = {}
        
        self._initialize_masks()
    
    def _initialize_masks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self.pruning_masks[name] = torch.ones_like(module.weight.data)
    
    def step(self):
        self.current_step += 1
        
        if self.current_step % self.pruning_frequency == 0:
            current_sparsity = self._compute_current_sparsity()
            self._update_masks(current_sparsity)
            self._apply_masks()
    
    def _compute_current_sparsity(self) -> float:
        if self.pruning_schedule == 'polynomial':
            progress = min(1.0, self.current_step / 10000)
            sparsity = self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3
        elif self.pruning_schedule == 'exponential':
            progress = min(1.0, self.current_step / 10000)
            sparsity = self.final_sparsity - (self.final_sparsity - self.initial_sparsity) * np.exp(-5 * progress)
        else:
            progress = min(1.0, self.current_step / 10000)
            sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress
        
        return sparsity
    
    def _update_masks(self, target_sparsity: float):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight.data
                current_mask = self.pruning_masks[name]
                
                weight_magnitude = torch.abs(weight)
                
                num_elements = weight.numel()
                num_to_prune = int(num_elements * target_sparsity)
                num_currently_pruned = (current_mask == 0).sum().item()
                
                if num_to_prune > num_currently_pruned:
                    additional_to_prune = num_to_prune - num_currently_pruned
                    
                    unpruned_weights = weight_magnitude * current_mask
                    unpruned_weights[current_mask == 0] = float('inf')
                    
                    _, indices_to_prune = torch.topk(
                        unpruned_weights.view(-1), additional_to_prune, largest=False
                    )
                    
                    flat_mask = current_mask.view(-1)
                    flat_mask[indices_to_prune] = 0
                    
                    self.pruning_masks[name] = flat_mask.view_as(current_mask)
    
    def _apply_masks(self):
        for name, module in self.model.named_modules():
            if name in self.pruning_masks:
                mask = self.pruning_masks[name]
                module.weight.data *= mask

class AutomaticPruning:
    def __init__(self,
                 model: nn.Module,
                 target_flops_reduction: float = 0.5,
                 accuracy_threshold: float = 0.02):
        
        self.model = model
        self.target_flops_reduction = target_flops_reduction
        self.accuracy_threshold = accuracy_threshold
        
        self.original_flops = self._compute_flops()
        self.original_accuracy = None
        
    def auto_prune(self,
                   train_dataloader: torch.utils.data.DataLoader,
                   val_dataloader: torch.utils.data.DataLoader,
                   device: torch.device) -> nn.Module:
        
        self.original_accuracy = self._evaluate_model(val_dataloader, device)
        
        pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        best_model = None
        best_ratio = 0.0
        
        for ratio in pruning_ratios:
            pruned_model = self._prune_model_with_ratio(ratio, train_dataloader, device)
            
            current_flops = self._compute_flops(pruned_model)
            flops_reduction = 1 - (current_flops / self.original_flops)
            
            if flops_reduction >= self.target_flops_reduction:
                accuracy = self._evaluate_model(val_dataloader, device, pruned_model)
                accuracy_drop = self.original_accuracy - accuracy
                
                if accuracy_drop <= self.accuracy_threshold:
                    best_model = pruned_model
                    best_ratio = ratio
                    break
        
        if best_model is None:
            print(f"Warning: Could not achieve target FLOPS reduction while maintaining accuracy")
            best_model = self.model
        
        return best_model
    
    def _prune_model_with_ratio(self,
                               ratio: float,
                               dataloader: torch.utils.data.DataLoader,
                               device: torch.device) -> nn.Module:
        
        import copy
        model_copy = copy.deepcopy(self.model)
        
        pruner = StructuredPruning(model_copy, {'pruning_ratio': ratio})
        pruner.compute_importance_scores(dataloader, device)
        pruner.generate_pruning_masks()
        pruner.apply_pruning()
        
        return model_copy
    
    def _compute_flops(self, model: nn.Module = None) -> int:
        if model is None:
            model = self.model
        
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
            elif isinstance(module, nn.Conv1d):
                kernel_flops = module.kernel_size[0] * module.in_channels
                output_elements = module.out_channels
                total_flops += kernel_flops * output_elements
            elif isinstance(module, nn.Conv2d):
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                output_elements = module.out_channels
                total_flops += kernel_flops * output_elements
        
        return total_flops
    
    def _evaluate_model(self,
                       dataloader: torch.utils.data.DataLoader,
                       device: torch.device,
                       model: nn.Module = None) -> float:
        
        if model is None:
            model = self.model
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                inputs, targets = _prepare_batch(batch_data, device)
                outputs = _forward_model(model, inputs)
                
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
                
                if targets is not None:
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                else:
                    raise ValueError("Evaluation targets are required for pruning benchmarks")
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

class PruningBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_pruning_methods(self,
                                 model: nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 device: torch.device) -> Dict[str, Dict[str, float]]:
        
        methods = {
            'magnitude': {'pruning_method': 'magnitude'},
            'gradient': {'pruning_method': 'gradient'},
            'fisher': {'pruning_method': 'fisher'},
            'taylor': {'pruning_method': 'taylor'}
        }
        
        results = {}
        
        for method_name, config in methods.items():
            import copy
            model_copy = copy.deepcopy(model)
            
            pruner = StructuredPruning(model_copy, config)
            pruner.compute_importance_scores(dataloader, device)
            pruner.generate_pruning_masks()
            pruner.apply_pruning()
            
            original_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
            pruned_params = sum(torch.count_nonzero(p).item() for p in model_copy.parameters())
            
            compression_ratio = original_params / pruned_params
            
            results[method_name] = {
                'compression_ratio': compression_ratio,
                'parameter_reduction': 1 - (pruned_params / original_params)
            }
        
        self.results = results
        return results
