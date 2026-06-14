from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeMiner(nn.Module):

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 64, **_: Dict):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Softmax keeps lambda_1 + lambda_2 + lambda_3 = 1.
        self.lambda_logits = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        positive_pairs: torch.Tensor,
        candidate_negatives: torch.Tensor,
        num_hard_negatives: int = 5,
    ) -> Dict[str, torch.Tensor]:
        positive_pairs = positive_pairs.reshape(positive_pairs.size(0), -1)[:, 0]
        positive = item_embeddings[positive_pairs]
        candidates = item_embeddings[candidate_negatives]

        user_norm = F.normalize(user_embeddings, dim=-1)
        positive_norm = F.normalize(positive, dim=-1)
        candidate_norm = F.normalize(candidates, dim=-1)
        user_negative = (user_norm.unsqueeze(1) * candidate_norm).sum(dim=-1)
        positive_negative = (positive_norm.unsqueeze(1) * candidate_norm).sum(dim=-1)
        user_positive = (user_norm * positive_norm).sum(dim=-1, keepdim=True)
        features = torch.stack(
            [user_negative, positive_negative, user_positive.expand_as(user_negative)], dim=-1
        )
        learned_difficulty = self.difficulty_estimator(features).squeeze(-1)
        lambdas = self.lambda_logits.softmax(dim=0)
        scores = (
            lambdas[0] * user_negative
            + lambdas[1] * positive_negative
            + lambdas[2] * learned_difficulty
        )
        count = min(num_hard_negatives, candidate_negatives.size(1))
        top_scores, top_indices = scores.topk(count, dim=1)
        hard_negatives = candidate_negatives.gather(1, top_indices)
        return {
            "hard_negatives": hard_negatives,
            "selection_scores": scores,
            "hard_negative_scores": top_scores,
            "difficulty_scores": learned_difficulty,
            "lambda_weights": lambdas,
            "similarities": {
                "user_negative": user_negative,
                "positive_negative": positive_negative,
                "user_positive": user_positive.squeeze(-1),
            },
        }


class DynamicNegativeSampler(nn.Module):

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        candidate_multiplier: int = 5,
        **_: Dict,
    ):
        super().__init__()
        self.num_items = num_items
        self.candidate_multiplier = candidate_multiplier
        self.hard_negative_miner = HardNegativeMiner(embedding_dim)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        positive_items: torch.Tensor,
        num_negatives: int = 10,
        hard_negative_ratio=0.4,
        user_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = user_embeddings.size(0)
        if torch.is_tensor(hard_negative_ratio):
            ratio = float(hard_negative_ratio.mean().detach().clamp(0, 1))
        else:
            ratio = min(max(float(hard_negative_ratio), 0.0), 1.0)
        num_hard = min(int(round(num_negatives * ratio)), num_negatives)
        num_random = num_negatives - num_hard
        candidates = self._sample_candidates(
            positive_items, user_history, max(num_negatives * self.candidate_multiplier, num_hard)
        )
        mining = self.hard_negative_miner(
            user_embeddings,
            item_embeddings,
            positive_items,
            candidates,
            num_hard_negatives=max(num_hard, 1),
        )
        hard = mining["hard_negatives"][:, :num_hard]
        random = candidates[:, :num_random]
        negatives = torch.cat([hard, random], dim=1)
        return {
            "negative_samples": negatives,
            "hard_negatives": hard,
            "random_negatives": random,
            "hard_negative_ratio": user_embeddings.new_tensor(ratio),
            **mining,
        }

    def _sample_candidates(
        self,
        positive_items: torch.Tensor,
        user_history: Optional[torch.Tensor],
        count: int,
    ) -> torch.Tensor:
        positive_items = positive_items.reshape(positive_items.size(0), -1)[:, 0]
        rows = []
        for batch_index, positive in enumerate(positive_items):
            excluded = {int(positive)}
            if user_history is not None:
                excluded.update(int(item) for item in user_history[batch_index].tolist() if int(item) >= 0)
            allowed = torch.tensor(
                [item for item in range(self.num_items) if item not in excluded],
                device=positive_items.device,
                dtype=torch.long,
            )
            if allowed.numel() == 0:
                raise ValueError("No valid negative items remain after excluding the user history")
            draws = torch.randint(0, allowed.numel(), (count,), device=positive_items.device)
            rows.append(allowed[draws])
        return torch.stack(rows)
