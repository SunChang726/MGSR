from itertools import combinations
from typing import Dict, Optional

import torch
import torch.nn as nn

from .cross_modal_attention import (
    MODALITIES,
    AdaptiveModalityWeightGenerator,
    ContrastiveModalityAlignment,
    CrossModalAttention,
)
from .quantizer import SemanticIDGenerator


class MultimodalEncoder(nn.Module):

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 768,
        audio_dim: int = 768,
        hidden_dim: int = 256,
        category_dim: int = 64,
        num_attention_heads: int = 8,
        coarse_vocab_size: int = 256,
        fine_vocab_size: int = 1024,
        alignment_temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.category_dim = category_dim
        self.cross_modal_attention = CrossModalAttention(
            text_dim=text_dim,
            vision_dim=vision_dim,
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.modality_weight_generator = AdaptiveModalityWeightGenerator(
            category_dim=category_dim
        )
        self.quality_estimators = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
                for name in MODALITIES
            }
        )
        self.alignment = ContrastiveModalityAlignment(alignment_temperature)
        self.semantic_id_generator = SemanticIDGenerator(
            input_dim=hidden_dim,
            coarse_vocab_size=coarse_vocab_size,
            fine_vocab_size=fine_vocab_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        category_features: Optional[torch.Tensor] = None,
        user_modality_preferences: Optional[torch.Tensor] = None,
        modality_qualities: Optional[torch.Tensor] = None,
        generate_semantic_ids: bool = True,
    ) -> Dict[str, torch.Tensor]:
        raw = {
            "text": text_features,
            "vision": vision_features,
            "audio": audio_features,
        }
        available = [name for name in MODALITIES if raw[name] is not None]
        if not available:
            raise ValueError("At least one pre-extracted modality feature must be provided")
        batch_size = raw[available[0]].size(0)
        device = raw[available[0]].device
        dtype = raw[available[0]].dtype

        availability_mask = torch.tensor(
            [name in available for name in MODALITIES], device=device
        ).unsqueeze(0).expand(batch_size, -1)
        if category_features is None:
            category_features = torch.zeros(batch_size, self.category_dim, device=device, dtype=dtype)
        if user_modality_preferences is None:
            user_modality_preferences = availability_mask.to(dtype)
            user_modality_preferences = user_modality_preferences / user_modality_preferences.sum(
                dim=-1, keepdim=True
            )
        if modality_qualities is None:
            projected = self.cross_modal_attention.project_modalities(
                text_features=text_features,
                vision_features=vision_features,
                audio_features=audio_features,
            )
            quality_columns = []
            for name in MODALITIES:
                if name in projected:
                    quality_columns.append(self.quality_estimators[name](projected[name]))
                else:
                    quality_columns.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))
            modality_qualities = torch.cat(quality_columns, dim=-1)

        modality_weights = self.modality_weight_generator(
            category_features=category_features,
            user_modality_preferences=user_modality_preferences,
            modality_qualities=modality_qualities,
            availability_mask=availability_mask,
        )
        attention = self.cross_modal_attention(
            text_features=text_features,
            vision_features=vision_features,
            audio_features=audio_features,
            modality_weights=modality_weights,
            return_details=True,
        )

        pair_losses = [
            self.alignment(
                attention["attended_features"][first],
                attention["attended_features"][second],
            )
            for first, second in combinations(available, 2)
        ]
        alignment_loss = (
            torch.stack(pair_losses).mean()
            if pair_losses
            else attention["fused_features"].new_zeros(())
        )

        results = {
            **attention,
            "alignment_loss": alignment_loss,
            "availability_mask": availability_mask,
            "attention_modality_weights": attention["modality_weights"],
            "modality_weights": modality_weights,
        }
        if generate_semantic_ids:
            semantic = self.semantic_id_generator(attention["fused_features"])
            results.update(semantic)
            results["quantization_loss"] = semantic["loss"]
        return results

    def encode_items(self, item_data: Dict[str, torch.Tensor], batch_size: int = 256):
        outputs = []
        count = next(iter(item_data.values())).size(0)
        for start in range(0, count, batch_size):
            batch = {key: value[start : start + batch_size] for key, value in item_data.items()}
            outputs.append(self.forward(**batch))
        keys = ("fused_features", "semantic_ids", "coarse_ids", "fine_ids")
        return {key: torch.cat([output[key] for output in outputs], dim=0) for key in keys}

    def get_modality_importance(
        self,
        category_features: torch.Tensor,
        user_modality_preferences: torch.Tensor,
        modality_qualities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.modality_weight_generator(
            category_features, user_modality_preferences, modality_qualities
        )


class LightweightMultimodalEncoder(MultimodalEncoder):
    def __init__(self, **kwargs):
        kwargs.setdefault("hidden_dim", 128)
        kwargs.setdefault("num_attention_heads", 4)
        super().__init__(**kwargs)


class IncrementalMultimodalEncoder(MultimodalEncoder):
    def __init__(self, memory_size: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.item_memory = {}

    def update_item_memory(self, item_ids, features: torch.Tensor):
        for item_id, feature in zip(item_ids, features.detach().cpu()):
            self.item_memory[item_id] = feature
            if len(self.item_memory) > self.memory_size:
                self.item_memory.pop(next(iter(self.item_memory)))

    def get_item_features(self, item_ids):
        if any(item_id not in self.item_memory for item_id in item_ids):
            return None
        return torch.stack([self.item_memory[item_id] for item_id in item_ids])

    def incremental_update(self, new_item_data: Dict[str, torch.Tensor], item_ids):
        outputs = self.encode_items(new_item_data)
        self.update_item_memory(item_ids, outputs["fused_features"])
        return outputs
