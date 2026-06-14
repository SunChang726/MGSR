from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRankingModule(nn.Module):

    def __init__(
        self,
        diversity_penalty: float = 0.1,
        novelty_balance: float = 0.8,
        diversity_reference_k: int = 20,
        **_: Dict,
    ):
        super().__init__()
        self.diversity_penalty = diversity_penalty
        self.novelty_balance = novelty_balance
        self.diversity_reference_k = diversity_reference_k

    def forward(
        self,
        conditional_user: torch.Tensor,
        candidate_fine_embeddings: torch.Tensor,
        history_fine_embeddings: Optional[torch.Tensor] = None,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        candidate = F.normalize(candidate_fine_embeddings, dim=-1)
        conditional = F.normalize(conditional_user, dim=-1)
        relevance = (conditional * candidate).sum(dim=-1)

        pairwise = candidate @ candidate.transpose(0, 1)
        reference_count = min(self.diversity_reference_k, candidate.size(0))
        reference = relevance.topk(reference_count).indices
        penalties = pairwise[:, reference].sum(dim=-1)
        is_reference = torch.isin(torch.arange(candidate.size(0), device=candidate.device), reference)
        penalties = penalties - is_reference.to(pairwise.dtype)
        diversity_score = relevance - self.diversity_penalty * penalties

        if history_fine_embeddings is None or history_fine_embeddings.numel() == 0:
            novelty = torch.ones_like(relevance)
        else:
            history = F.normalize(history_fine_embeddings, dim=-1)
            novelty = 1.0 - (candidate @ history.transpose(0, 1)).max(dim=-1).values
        final = self.novelty_balance * diversity_score + (1 - self.novelty_balance) * novelty
        return {
            "final_scores": final,
            "relevance_scores": relevance,
            "diversity_scores": diversity_score,
            "diversity_penalty": penalties,
            "novelty_scores": novelty,
        }

    def rerank_with_mmr(
        self, scores: torch.Tensor, item_embeddings: torch.Tensor, lambda_param: float = 0.5, top_k: int = 20
    ):
        # Optional extension retained for comparisons; the paper path uses Eq. (34).
        embeddings = F.normalize(item_embeddings, dim=-1)
        selected = []
        remaining = list(range(scores.numel()))
        while remaining and len(selected) < top_k:
            values = []
            for index in remaining:
                redundancy = (
                    (embeddings[index] @ embeddings[selected].transpose(0, 1)).max()
                    if selected
                    else scores.new_zeros(())
                )
                values.append(lambda_param * scores[index] - (1 - lambda_param) * redundancy)
            chosen = remaining[int(torch.stack(values).argmax())]
            selected.append(chosen)
            remaining.remove(chosen)
        return {"reranked_items": selected, "reranked_scores": scores[selected]}


class MultiObjectiveRanker(nn.Module):

    def __init__(self, feature_dim: int = 256, num_objectives: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) for _ in range(num_objectives)]
        )
        self.weights = nn.Parameter(torch.zeros(num_objectives))

    def forward(self, interaction_features: torch.Tensor, **_: Dict):
        objective_scores = torch.cat([head(interaction_features) for head in self.heads], dim=-1)
        weights = self.weights.softmax(dim=0)
        return {"final_scores": (objective_scores * weights).sum(dim=-1), "objective_scores": objective_scores}


class PersonalizedRankingLoss(nn.Module):
    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor, **_: Dict):
        return F.softplus(negative_scores - positive_scores.unsqueeze(-1)).mean()


class ListwiseRankingLoss(nn.Module):
    def forward(self, scores: torch.Tensor, relevance_labels: torch.Tensor):
        return -(relevance_labels.float().softmax(dim=-1) * scores.log_softmax(dim=-1)).sum(dim=-1).mean()


class RankingEvaluator:
    def evaluate_ranking_quality(self, predicted_rankings, ground_truth, k_values=(5, 10, 20)):
        metrics = {}
        for k in k_values:
            hits = [float(target in ranking[:k]) for ranking, target in zip(predicted_rankings, ground_truth)]
            metrics[f"HR@{k}"] = sum(hits) / max(len(hits), 1)
        return metrics


class InteractionLayer(nn.Identity):
    pass


class ContextualBanditsRanker(MultiObjectiveRanker):
    pass
