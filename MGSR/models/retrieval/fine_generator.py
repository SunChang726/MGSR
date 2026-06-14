from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineGrainedGenerator(nn.Module):

    def __init__(
        self,
        user_embedding_dim: int = 256,
        hidden_dim: int = 256,
        coarse_vocab_size: int = 256,
        fine_vocab_size: int = 1024,
        eta_fine: float = 0.1,
        **_: Dict,
    ):
        super().__init__()
        self.coarse_vocab_size = coarse_vocab_size
        self.fine_vocab_size = fine_vocab_size
        self.eta_fine = eta_fine
        self.coarse_embedding = nn.Embedding(coarse_vocab_size, hidden_dim)
        self.fine_embedding = nn.Embedding(fine_vocab_size, hidden_dim)
        self.conditional_projection = nn.Sequential(
            nn.Linear(user_embedding_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output_weights = nn.Parameter(
            torch.empty(coarse_vocab_size, fine_vocab_size, hidden_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(coarse_vocab_size, fine_vocab_size))
        nn.init.xavier_uniform_(self.output_weights)

    def conditional_user(
        self, user_embeddings: torch.Tensor, coarse_ids: torch.Tensor
    ) -> torch.Tensor:
        coarse = self.coarse_embedding(coarse_ids)
        if coarse_ids.dim() == 1:
            return self.conditional_projection(torch.cat([user_embeddings, coarse], dim=-1))
        user = user_embeddings.unsqueeze(1).expand(-1, coarse_ids.size(1), -1)
        return self.conditional_projection(torch.cat([user, coarse], dim=-1))

    def logits_for_codes(
        self, conditional_user: torch.Tensor, coarse_ids: torch.Tensor
    ) -> torch.Tensor:
        weights = self.output_weights[coarse_ids]
        bias = self.output_bias[coarse_ids]
        return torch.einsum("...h,...fh->...f", conditional_user, weights) + bias

    def forward(
        self,
        user_embeddings: torch.Tensor,
        coarse_ids: torch.Tensor,
        target_fine_ids: Optional[torch.Tensor] = None,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        conditional = self.conditional_user(user_embeddings, coarse_ids)
        logits = self.logits_for_codes(conditional, coarse_ids)
        probabilities = logits.softmax(dim=-1)
        loss = None
        if target_fine_ids is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.fine_vocab_size), target_fine_ids.reshape(-1))
        return {
            "conditional_user": conditional,
            "logits": logits,
            "probabilities": probabilities,
            "loss": loss,
        }

    def score_items(
        self,
        user_embedding: torch.Tensor,
        candidate_coarse_ids: torch.Tensor,
        candidate_fine_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        repeated_user = user_embedding.expand(candidate_coarse_ids.size(0), -1)
        conditional = self.conditional_user(repeated_user, candidate_coarse_ids)
        logits = self.logits_for_codes(conditional, candidate_coarse_ids)
        probabilities = logits.softmax(dim=-1)
        predicted = probabilities.gather(1, candidate_fine_ids.unsqueeze(1)).squeeze(1)
        fine = F.normalize(self.fine_embedding(candidate_fine_ids), dim=-1)
        compatibility = (F.normalize(conditional, dim=-1) * fine).sum(dim=-1)
        return {
            "scores": predicted + self.eta_fine * compatibility,
            "conditional_user": conditional,
            "fine_embeddings": fine,
            "probabilities": probabilities,
        }


class FineGrainedRetriever(nn.Module):
    def __init__(self, top_l: int = 50, **_: Dict):
        super().__init__()
        self.top_l = top_l

    def forward(self, candidate_items: torch.Tensor, scores: torch.Tensor, top_l: Optional[int] = None, **_: Dict):
        top_l = top_l or self.top_l
        values, order = scores.topk(min(top_l, candidate_items.numel()))
        return {"ranked_items": candidate_items[order], "ranking_scores": values}


class FineGrainedTransformerLayer(nn.TransformerEncoderLayer):

    pass
