from typing import Dict

import torch
import torch.nn as nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class UserBehaviorProfiler(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        max_sequence_length: int = 200,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        # f_u: 3, t_u: 3, p_u: 3, a_u: 1 (Eq. 16).
        self.profile_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, user_sequences: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        item_ids = user_sequences["item_ids"]
        mask = user_sequences.get("sequence_mask", item_ids.ge(0)).bool()
        timestamps = user_sequences.get(
            "timestamps", torch.zeros_like(item_ids, dtype=torch.float)
        ).float()
        feedback = user_sequences.get(
            "feedback_strength", torch.ones_like(item_ids, dtype=torch.float)
        ).float()

        lengths = mask.sum(dim=1).float()
        activity = torch.stack(
            [
                lengths / max(float(self.max_sequence_length), 1.0),
                torch.log1p(lengths),
                _masked_mean(feedback, mask, dim=1),
            ],
            dim=-1,
        )

        intervals = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(0)
        interval_mask = mask[:, 1:] & mask[:, :-1]
        interval_mean = _masked_mean(intervals, interval_mask, dim=1)
        centered = (intervals - interval_mean.unsqueeze(1)).square()
        interval_std = _masked_mean(centered, interval_mask, dim=1).sqrt()
        latest = timestamps.masked_fill(~mask, float("-inf")).max(dim=1).values
        latest = torch.where(torch.isfinite(latest), latest, torch.zeros_like(latest))
        temporal = torch.stack(
            [torch.log1p(latest.clamp_min(0)), torch.log1p(interval_mean), torch.log1p(interval_std)],
            dim=-1,
        )

        modality = user_sequences.get("modality_preferences")
        if modality is None:
            modality = user_sequences.get("modality_mask")
        if modality is None:
            modality_preference = torch.full(
                (item_ids.size(0), 3), 1.0 / 3, device=item_ids.device
            )
        elif modality.dim() == 3:
            modality_preference = _masked_mean(modality.float(), mask, dim=1)
        else:
            modality_preference = modality.float()
        modality_preference = modality_preference / modality_preference.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        category_ids = user_sequences.get("category_ids")
        if category_ids is None:
            category_ids = item_ids
        entropy = self._category_entropy(category_ids, mask).unsqueeze(-1)

        statistics = torch.cat([activity, temporal, modality_preference, entropy], dim=-1)
        profile = self.profile_encoder(statistics)
        return {
            "user_profile": profile,
            "activity_features": activity,
            "temporal_features": temporal,
            "modality_features": modality_preference,
            "interest_stability": entropy,
            "statistics": statistics,
        }

    @staticmethod
    def _category_entropy(category_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        entropies = []
        for categories, valid in zip(category_ids, mask):
            selected = categories[valid]
            if selected.numel() == 0:
                entropies.append(categories.new_zeros((), dtype=torch.float))
                continue
            _, counts = torch.unique(selected, return_counts=True)
            probabilities = counts.float() / counts.sum()
            entropies.append(-(probabilities * probabilities.clamp_min(1e-8).log()).sum())
        return torch.stack(entropies)


class BehaviorTypeClassifier:

    @staticmethod
    def classify_behavior(user_profile: torch.Tensor, behavior_logits: torch.Tensor):
        probabilities = behavior_logits.softmax(dim=-1)
        return {
            "behavior_types": probabilities.argmax(dim=-1),
            "behavior_probs": probabilities,
            "confidence": probabilities.max(dim=-1).values,
        }


class AdaptiveBehaviorAnalyzer(nn.Module):

    def __init__(self, profile_dim: int = 64, num_clusters: int = 8):
        super().__init__()
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, profile_dim))

    def forward(self, user_profiles: torch.Tensor):
        distances = torch.cdist(user_profiles, self.cluster_centers)
        return {
            "cluster_assignments": distances.argmin(dim=-1),
            "cluster_probs": (-distances).softmax(dim=-1),
            "cluster_distances": distances,
        }
