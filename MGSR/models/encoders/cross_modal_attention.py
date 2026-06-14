import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITIES = ("text", "vision", "audio")


class CrossModalAttention(nn.Module):

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 768,
        audio_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleDict(
            {
                "text": nn.Linear(text_dim, hidden_dim),
                "vision": nn.Linear(vision_dim, hidden_dim),
                "audio": nn.Linear(audio_dim, hidden_dim),
            }
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def project_modalities(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raw = {
            "text": text_features,
            "vision": vision_features,
            "audio": audio_features,
        }
        return {
            name: self.projections[name](features)
            for name, features in raw.items()
            if features is not None
        }

    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        modality_weights: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        projected = self.project_modalities(
            text_features=text_features,
            vision_features=vision_features,
            audio_features=audio_features,
        )
        if not projected:
            raise ValueError("At least one modality must be provided")

        available = [name for name in MODALITIES if name in projected]
        sequence = torch.stack([projected[name] for name in available], dim=1)
        attended, attention_weights = self.attention(
            sequence, sequence, sequence, need_weights=True, average_attn_weights=False
        )
        attended = self.layer_norm(
            sequence + self.dropout(self.output_projection(attended))
        )

        if modality_weights is None:
            weights = torch.full(
                (sequence.size(0), len(available)),
                1.0 / len(available),
                device=sequence.device,
                dtype=sequence.dtype,
            )
        else:
            if modality_weights.size(-1) == len(MODALITIES):
                indices = [MODALITIES.index(name) for name in available]
                weights = modality_weights[:, indices]
            elif modality_weights.size(-1) == len(available):
                weights = modality_weights
            else:
                raise ValueError("modality_weights has an incompatible last dimension")
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        fused = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        if not return_details:
            return fused

        return {
            "fused_features": fused,
            "projected_features": projected,
            "attended_features": {
                name: attended[:, index] for index, name in enumerate(available)
            },
            "attention_weights": attention_weights,
            "modality_weights": weights,
            "available_modalities": available,
        }


class AdaptiveModalityWeightGenerator(nn.Module):

    def __init__(
        self,
        category_dim: int = 64,
        user_preference_dim: int = 3,
        hidden_dim: int = 128,
        num_modalities: int = 3,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.quality_estimator = nn.Sequential(
            nn.Linear(category_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid(),
        )
        self.weight_generator = nn.Sequential(
            nn.Linear(category_dim + user_preference_dim + num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
        )

    def forward(
        self,
        category_features: torch.Tensor,
        user_modality_preferences: torch.Tensor,
        modality_qualities: Optional[torch.Tensor] = None,
        availability_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if modality_qualities is None:
            modality_qualities = self.quality_estimator(category_features)
        inputs = torch.cat(
            [category_features, user_modality_preferences, modality_qualities], dim=-1
        )
        logits = self.weight_generator(inputs)
        if availability_mask is not None:
            logits = logits.masked_fill(~availability_mask.bool(), float("-inf"))
        return F.softmax(logits, dim=-1)


class ContrastiveModalityAlignment(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = F.normalize(first, dim=-1)
        second = F.normalize(second, dim=-1)
        logits = first @ second.transpose(0, 1) / self.temperature
        labels = torch.arange(first.size(0), device=first.device)
        return F.cross_entropy(logits, labels)


class HierarchicalCrossModalFusion(nn.Module):

    def __init__(self, num_layers: int = 1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([CrossModalAttention(**kwargs) for _ in range(num_layers)])

    def forward(self, **kwargs) -> torch.Tensor:
        # The paper applies attention to the short modality sequence once. Extra layers
        # are retained only as an optional extension and reuse the original modalities.
        output = None
        for layer in self.layers:
            output = layer(**kwargs)
        return output
