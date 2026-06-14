from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseGrainedGenerator(nn.Module):

    def __init__(
        self,
        user_embedding_dim: int = 256,
        hidden_dim: int = 256,
        coarse_vocab_size: int = 256,
        eta_coarse: float = 0.1,
        **_: Dict,
    ):
        super().__init__()
        self.coarse_vocab_size = coarse_vocab_size
        self.eta_coarse = eta_coarse
        self.user_projection = nn.Sequential(
            nn.Linear(user_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.coarse_embedding = nn.Embedding(coarse_vocab_size, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, coarse_vocab_size)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        target_coarse_ids: Optional[torch.Tensor] = None,
        top_n: int = 10,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        coarse_user = self.user_projection(user_embeddings)
        logits = self.output_projection(coarse_user)
        probabilities = logits.softmax(dim=-1)
        top_probabilities, top_codes = probabilities.topk(
            min(top_n, self.coarse_vocab_size), dim=-1
        )
        loss = None
        if target_coarse_ids is not None:
            loss = F.cross_entropy(logits, target_coarse_ids.reshape(-1))
        return {
            "coarse_user": coarse_user,
            "logits": logits,
            "probabilities": probabilities,
            "top_codes": top_codes,
            "top_probabilities": top_probabilities,
            "loss": loss,
        }

    def score_items(
        self,
        coarse_user: torch.Tensor,
        probabilities: torch.Tensor,
        item_coarse_ids: torch.Tensor,
    ) -> torch.Tensor:
        code_embeddings = F.normalize(self.coarse_embedding(item_coarse_ids), dim=-1)
        compatibility = F.normalize(coarse_user, dim=-1) @ code_embeddings.transpose(0, 1)
        probability_scores = probabilities[:, item_coarse_ids]
        return probability_scores + self.eta_coarse * compatibility


class CoarseGrainedRetriever(nn.Module):

    def __init__(self, top_n: int = 10, top_m: int = 10, **_: Dict):
        super().__init__()
        self.top_n = top_n
        self.top_m = top_m

    def forward(
        self,
        top_codes: torch.Tensor,
        item_coarse_ids: torch.Tensor,
        item_scores: torch.Tensor,
        top_m: Optional[int] = None,
        **_: Dict,
    ):
        top_m = top_m or self.top_m
        candidate_items = []
        candidate_scores = []
        for batch_index, codes in enumerate(top_codes):
            selected_items = []
            selected_scores = []
            for code in codes:
                indices = torch.where(item_coarse_ids == code)[0]
                if indices.numel() == 0:
                    continue
                scores = item_scores[batch_index, indices]
                values, order = scores.topk(min(top_m, indices.numel()))
                selected_items.append(indices[order])
                selected_scores.append(values)
            if selected_items:
                candidate_items.append(torch.cat(selected_items))
                candidate_scores.append(torch.cat(selected_scores))
            else:
                candidate_items.append(torch.empty(0, dtype=torch.long, device=top_codes.device))
                candidate_scores.append(torch.empty(0, device=top_codes.device))
        return {"candidate_items": candidate_items, "candidate_scores": candidate_scores}


class TransformerDecoderLayer(nn.TransformerEncoderLayer):

    pass
