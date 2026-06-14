from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class AdaptiveStrategyGenerator(nn.Module):

    def __init__(
        self,
        user_profile_dim: int = 64,
        hidden_dim: int = 128,
        temperature_range: Tuple[float, float] = (0.01, 0.5),
        target_hard_negative_ratio: float = 0.4,
        curriculum_steps: int = 10000,
        **_: Dict,
    ):
        super().__init__()
        self.temperature_min, self.temperature_max = temperature_range
        self.target_hard_negative_ratio = target_hard_negative_ratio
        self.curriculum_steps = max(curriculum_steps, 1)
        self.temperature_mlp = nn.Sequential(
            nn.Linear(user_profile_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        user_profiles: torch.Tensor,
        item_features: Optional[torch.Tensor] = None,
        training_step: int = 0,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        normalized = torch.sigmoid(self.temperature_mlp(user_profiles))
        temperature = self.temperature_min + (
            self.temperature_max - self.temperature_min
        ) * normalized
        progress = min(max(float(training_step) / self.curriculum_steps, 0.0), 1.0)
        hard_ratio = user_profiles.new_full(
            (user_profiles.size(0),), self.target_hard_negative_ratio * progress
        )
        return {
            "temperature": temperature.squeeze(-1),
            "hard_negative_ratio": hard_ratio,
            "curriculum_progress": user_profiles.new_tensor(progress),
        }

    def generate_negative_sampling_strategy(
        self, user_profiles: torch.Tensor, item_features: Optional[torch.Tensor] = None, training_step: int = 0
    ):
        return self.forward(user_profiles, item_features, training_step)


class DataAugmentationEngine:

    def apply_augmentation(self, sequences: torch.Tensor, strategy: str, intensity: float):
        if strategy != "sequence_masking":
            return sequences
        output = sequences.clone()
        valid = output.ne(0)
        mask = torch.rand_like(output.float()).lt(float(intensity)) & valid
        output[mask] = 0
        return output


class NegativeSamplingEngine:

    def __init__(self, num_items: int = 10000):
        self.num_items = num_items

    def sample_negatives(
        self, positive_items: torch.Tensor, strategy: str, params: Dict[str, float], num_negatives: int = 10
    ):
        samples = torch.randint(
            0, self.num_items, (positive_items.size(0), num_negatives), device=positive_items.device
        )
        return samples
