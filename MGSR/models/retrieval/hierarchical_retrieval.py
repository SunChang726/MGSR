from typing import Dict, Optional

import torch
import torch.nn as nn

from .coarse_generator import CoarseGrainedGenerator, CoarseGrainedRetriever
from .fine_generator import FineGrainedGenerator, FineGrainedRetriever
from .ranking_module import HierarchicalRankingModule, RankingEvaluator


class HierarchicalGenerativeRetrieval(nn.Module):

    def __init__(
        self,
        user_embedding_dim: int = 256,
        hidden_dim: int = 256,
        coarse_vocab_size: int = 256,
        fine_vocab_size: int = 1024,
        top_n_coarse: int = 10,
        top_m_per_coarse: int = 10,
        top_l_fine: int = 50,
        final_top_k: int = 20,
        eta_coarse: float = 0.1,
        eta_fine: float = 0.1,
        diversity_penalty: float = 0.1,
        novelty_balance: float = 0.8,
        **_: Dict,
    ):
        super().__init__()
        self.coarse_vocab_size = coarse_vocab_size
        self.fine_vocab_size = fine_vocab_size
        self.top_n_coarse = top_n_coarse
        self.top_m_per_coarse = top_m_per_coarse
        self.top_l_fine = top_l_fine
        self.final_top_k = final_top_k
        self.coarse_generator = CoarseGrainedGenerator(
            user_embedding_dim, hidden_dim, coarse_vocab_size, eta_coarse
        )
        self.coarse_retriever = CoarseGrainedRetriever(top_n_coarse, top_m_per_coarse)
        self.fine_generator = FineGrainedGenerator(
            user_embedding_dim, hidden_dim, coarse_vocab_size, fine_vocab_size, eta_fine
        )
        self.fine_retriever = FineGrainedRetriever(top_l_fine)
        self.ranking_module = HierarchicalRankingModule(
            diversity_penalty, novelty_balance, final_top_k
        )

    def forward(
        self,
        user_embeddings: torch.Tensor,
        catalog_semantic_ids: torch.Tensor,
        target_items: Optional[torch.Tensor] = None,
        history_item_ids: Optional[torch.Tensor] = None,
        **_: Dict,
    ) -> Dict:
        if catalog_semantic_ids.dim() != 2 or catalog_semantic_ids.size(1) != 2:
            raise ValueError("catalog_semantic_ids must have shape [num_items, 2]")
        item_coarse = catalog_semantic_ids[:, 0].long()
        item_fine = catalog_semantic_ids[:, 1].long()
        target_coarse = item_coarse[target_items] if target_items is not None else None
        target_fine = item_fine[target_items] if target_items is not None else None
        coarse = self.coarse_generator(
            user_embeddings, target_coarse_ids=target_coarse, top_n=self.top_n_coarse
        )
        all_coarse_scores = self.coarse_generator.score_items(
            coarse["coarse_user"], coarse["probabilities"], item_coarse
        )
        coarse_candidates = self.coarse_retriever(
            coarse["top_codes"], item_coarse, all_coarse_scores, self.top_m_per_coarse
        )

        final_rankings = []
        fine_outputs = []
        ranking_outputs = []
        for batch_index, candidates in enumerate(coarse_candidates["candidate_items"]):
            if candidates.numel() == 0:
                final_rankings.append([])
                fine_outputs.append({})
                ranking_outputs.append({})
                continue
            fine = self.fine_generator.score_items(
                user_embeddings[batch_index : batch_index + 1],
                item_coarse[candidates],
                item_fine[candidates],
            )
            filtered = self.fine_retriever(candidates, fine["scores"], self.top_l_fine)
            selected = filtered["ranked_items"]
            # Recompute conditional representations for the selected fine candidates.
            selected_fine = self.fine_generator.score_items(
                user_embeddings[batch_index : batch_index + 1],
                item_coarse[selected],
                item_fine[selected],
            )
            history_embeddings = None
            if history_item_ids is not None:
                valid_history = history_item_ids[batch_index]
                valid_history = valid_history[(valid_history >= 0) & (valid_history < catalog_semantic_ids.size(0))]
                if valid_history.numel():
                    history_embeddings = self.fine_generator.fine_embedding(item_fine[valid_history])
            ranking = self.ranking_module(
                selected_fine["conditional_user"],
                self.fine_generator.fine_embedding(item_fine[selected]),
                history_embeddings,
            )
            order = ranking["final_scores"].argsort(descending=True)[: self.final_top_k]
            final_rankings.append(selected[order].detach().cpu().tolist())
            fine_outputs.append({**fine, **filtered})
            ranking_outputs.append(ranking)

        fine_loss = user_embeddings.new_zeros(())
        if target_items is not None:
            target_result = self.fine_generator(
                user_embeddings, target_coarse, target_fine_ids=target_fine
            )
            fine_loss = target_result["loss"]
        coarse_loss = coarse["loss"] if coarse["loss"] is not None else user_embeddings.new_zeros(())
        return {
            "final_rankings": final_rankings,
            "coarse_results": coarse,
            "coarse_retrieval": coarse_candidates,
            "fine_results": fine_outputs,
            "ranking_results": ranking_outputs,
            "coarse_loss": coarse_loss,
            "fine_loss": fine_loss,
            "total_loss": coarse_loss + fine_loss,
        }


class HierarchicalRetrievalTrainer:

    def __init__(self, model: HierarchicalGenerativeRetrieval, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.evaluator = RankingEvaluator()

    def train_step(self, batch_data: Dict[str, torch.Tensor], target_items: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(target_items=target_items.to(self.device), **batch_data)
        output["total_loss"].backward()
        self.optimizer.step()
        return {"total_loss": output["total_loss"].item()}
