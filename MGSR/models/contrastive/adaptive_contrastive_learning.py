from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_strategy_generator import AdaptiveStrategyGenerator
from .hard_negative_mining import DynamicNegativeSampler
from .user_behavior_profiler import UserBehaviorProfiler


class AdaptiveContrastiveLearning(nn.Module):

    def __init__(
        self,
        user_profile_dim: int = 64,
        embedding_dim: int = 256,
        num_items: int = 10000,
        temperature_range: Tuple[float, float] = (0.01, 0.5),
        hard_negative_ratio: float = 0.4,
        num_negatives: int = 10,
        curriculum_steps: int = 10000,
        **_: Dict,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_behavior_profiler = UserBehaviorProfiler(embedding_dim=user_profile_dim)
        self.strategy_generator = AdaptiveStrategyGenerator(
            user_profile_dim=user_profile_dim,
            temperature_range=temperature_range,
            target_hard_negative_ratio=hard_negative_ratio,
            curriculum_steps=curriculum_steps,
        )
        self.negative_sampler = DynamicNegativeSampler(num_items, embedding_dim)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        user_sequences: Dict[str, torch.Tensor],
        positive_items: torch.Tensor,
        training_step: int = 0,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        profiles = self.user_behavior_profiler(user_sequences)
        strategy = self.strategy_generator(
            profiles["user_profile"], training_step=training_step
        )
        negatives = self.negative_sampler(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            positive_items=positive_items,
            num_negatives=self.num_negatives,
            hard_negative_ratio=strategy["hard_negative_ratio"],
            user_history=user_sequences.get("item_ids"),
        )

        positive_ids = positive_items.reshape(positive_items.size(0), -1)[:, 0]
        user = F.normalize(user_embeddings, dim=-1)
        positive = F.normalize(item_embeddings[positive_ids], dim=-1)
        negative = F.normalize(item_embeddings[negatives["negative_samples"]], dim=-1)
        temperature = strategy["temperature"].clamp_min(1e-6)
        positive_logits = (user * positive).sum(dim=-1) / temperature
        negative_logits = (user.unsqueeze(1) * negative).sum(dim=-1) / temperature.unsqueeze(1)
        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        labels = torch.zeros(user.size(0), dtype=torch.long, device=user.device)
        loss = F.cross_entropy(logits, labels)
        return {
            "total_loss": loss,
            "contrastive_loss": loss,
            "temperatures": temperature,
            "negative_samples": negatives["negative_samples"],
            "negative_sampling": negatives,
            "user_profiles": profiles,
            "adaptive_strategy": strategy,
            "logits": logits,
        }

    def get_adaptive_parameters(self):
        return {
            "temperature_range": torch.tensor(
                [self.strategy_generator.temperature_min, self.strategy_generator.temperature_max]
            ),
            "lambda_weights": self.negative_sampler.hard_negative_miner.lambda_logits.softmax(dim=0),
        }


class ContrastiveLearningEvaluator:
    def evaluate_contrastive_quality(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        positive_pairs: torch.Tensor,
        negative_pairs: torch.Tensor,
    ):
        user = F.normalize(user_embeddings, dim=-1)
        items = F.normalize(item_embeddings, dim=-1)
        positive = (user * items[positive_pairs]).sum(dim=-1)
        negative = (user.unsqueeze(1) * items[negative_pairs]).sum(dim=-1)
        return {
            "positive_similarity_mean": positive.mean().item(),
            "negative_similarity_mean": negative.mean().item(),
            "similarity_separation": (positive.mean() - negative.mean()).item(),
        }
