from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive.adaptive_contrastive_learning import AdaptiveContrastiveLearning
from .encoders.multimodal_encoder import MultimodalEncoder
from .retrieval.hierarchical_retrieval import HierarchicalGenerativeRetrieval


class MGSRModel(nn.Module):

    def __init__(
        self,
        num_items: int = 10000,
        text_feature_dim: int = 768,
        vision_feature_dim: int = 768,
        audio_feature_dim: int = 768,
        multimodal_embedding_dim: int = 256,
        hidden_dim: int = 768,
        category_feature_dim: int = 64,
        coarse_vocab_size: int = 256,
        fine_vocab_size: int = 1024,
        num_attention_heads: int = 8,
        num_sequence_layers: int = 3,
        max_sequence_length: int = 200,
        num_time_buckets: int = 128,
        dropout: float = 0.1,
        temperature_range=(0.01, 0.5),
        hard_negative_ratio: float = 0.4,
        top_n_coarse: int = 10,
        top_m_per_coarse: int = 10,
        top_l_fine: int = 50,
        final_top_k: int = 20,
        loss_weights=(1.0, 0.5, 0.3),
        **_: Dict,
    ):
        super().__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.num_time_buckets = num_time_buckets
        self.loss_weights = loss_weights

        self.multimodal_encoder = MultimodalEncoder(
            text_dim=text_feature_dim,
            vision_dim=vision_feature_dim,
            audio_dim=audio_feature_dim,
            hidden_dim=multimodal_embedding_dim,
            category_dim=category_feature_dim,
            num_attention_heads=num_attention_heads,
            coarse_vocab_size=coarse_vocab_size,
            fine_vocab_size=fine_vocab_size,
            dropout=dropout,
        )

        semantic_dim = multimodal_embedding_dim
        self.coarse_semantic_embedding = nn.Embedding(coarse_vocab_size, semantic_dim)
        self.fine_semantic_embedding = nn.Embedding(fine_vocab_size, semantic_dim)
        self.semantic_aggregation = nn.Sequential(
            nn.Linear(semantic_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        self.time_interval_embedding = nn.Embedding(num_time_buckets, hidden_dim)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(sequence_layer, num_sequence_layers)

        self.contrastive_learning = AdaptiveContrastiveLearning(
            embedding_dim=hidden_dim,
            num_items=num_items,
            temperature_range=temperature_range,
            hard_negative_ratio=hard_negative_ratio,
        )
        self.hierarchical_retrieval = HierarchicalGenerativeRetrieval(
            user_embedding_dim=hidden_dim,
            hidden_dim=hidden_dim,
            coarse_vocab_size=coarse_vocab_size,
            fine_vocab_size=fine_vocab_size,
            top_n_coarse=top_n_coarse,
            top_m_per_coarse=top_m_per_coarse,
            top_l_fine=top_l_fine,
            final_top_k=final_top_k,
        )

    def semantic_item_representations(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        coarse = self.coarse_semantic_embedding(semantic_ids[..., 0].long())
        fine = self.fine_semantic_embedding(semantic_ids[..., 1].long())
        return self.semantic_aggregation(torch.cat([coarse, fine], dim=-1))

    def encode_user_sequence(
        self, user_sequences: Dict[str, torch.Tensor], catalog_semantic_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        item_ids = user_sequences["item_ids"].long()
        sequence_mask = user_sequences.get("sequence_mask", item_ids.ge(0)).bool()
        safe_item_ids = item_ids.clamp(min=0, max=catalog_semantic_ids.size(0) - 1)
        semantic_ids = catalog_semantic_ids[safe_item_ids]
        values = self.semantic_item_representations(semantic_ids)

        positions = torch.arange(item_ids.size(1), device=item_ids.device)
        positions = positions.unsqueeze(0).expand_as(item_ids)
        timestamps = user_sequences.get(
            "timestamps", torch.zeros_like(item_ids, dtype=torch.float)
        ).float()
        intervals = torch.zeros_like(timestamps)
        intervals[:, 1:] = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(0)
        interval_buckets = torch.floor(torch.log2(intervals + 1)).long()
        interval_buckets = interval_buckets.clamp(max=self.num_time_buckets - 1)
        inputs = (
            values
            + self.position_embedding(positions)
            + self.time_interval_embedding(interval_buckets)
        )
        encoded = self.sequence_encoder(inputs, src_key_padding_mask=~sequence_mask)
        last_index = sequence_mask.sum(dim=1).clamp_min(1) - 1
        user_embedding = encoded[
            torch.arange(encoded.size(0), device=encoded.device), last_index
        ]
        return {
            "user_embeddings": user_embedding,
            "sequence_embeddings": encoded,
            "sequence_mask": sequence_mask,
            "sequence_semantic_ids": semantic_ids,
        }

    def forward(
        self,
        user_sequences: Dict[str, torch.Tensor],
        catalog_semantic_ids: Optional[torch.Tensor] = None,
        target_items: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        category_features: Optional[torch.Tensor] = None,
        user_modality_preferences: Optional[torch.Tensor] = None,
        modality_qualities: Optional[torch.Tensor] = None,
        training_step: int = 0,
        mode: str = "train",
        **_: Dict,
    ) -> Dict[str, Any]:
        multimodal_outputs = None
        if any(feature is not None for feature in (text_features, vision_features, audio_features)):
            multimodal_outputs = self.multimodal_encoder(
                text_features=text_features,
                vision_features=vision_features,
                audio_features=audio_features,
                category_features=category_features,
                user_modality_preferences=user_modality_preferences,
                modality_qualities=modality_qualities,
            )
            if catalog_semantic_ids is None:
                catalog_semantic_ids = multimodal_outputs["semantic_ids"]
        if catalog_semantic_ids is None:
            raise ValueError(
                "catalog_semantic_ids are required unless features for the entire catalog are provided"
            )
        if catalog_semantic_ids.size(0) != self.num_items:
            raise ValueError(
                f"Expected semantic IDs for {self.num_items} catalog items, got {catalog_semantic_ids.size(0)}"
            )

        user_outputs = self.encode_user_sequence(user_sequences, catalog_semantic_ids)
        user_embeddings = user_outputs["user_embeddings"]
        item_embeddings = self.semantic_item_representations(catalog_semantic_ids)
        retrieval = self.hierarchical_retrieval(
            user_embeddings=user_embeddings,
            catalog_semantic_ids=catalog_semantic_ids,
            target_items=target_items,
            history_item_ids=user_sequences["item_ids"],
        )

        zero = user_embeddings.new_zeros(())
        contrastive = {"total_loss": zero}
        if target_items is not None and mode == "train":
            contrastive = self.contrastive_learning(
                user_embeddings=user_embeddings,
                item_embeddings=item_embeddings,
                user_sequences=user_sequences,
                positive_items=target_items,
                training_step=training_step,
            )
        tokenization_loss = zero
        if multimodal_outputs is not None:
            tokenization_loss = (
                multimodal_outputs["alignment_loss"] + multimodal_outputs["quantization_loss"]
            )
        retrieval_loss = retrieval["total_loss"]
        total_loss = (
            self.loss_weights[0] * retrieval_loss
            + self.loss_weights[1] * contrastive["total_loss"]
            + self.loss_weights[2] * tokenization_loss
        )
        return {
            "total_loss": total_loss,
            "retrieval_loss": retrieval_loss,
            "contrastive_loss": contrastive["total_loss"],
            "tokenization_loss": tokenization_loss,
            "final_rankings": retrieval["final_rankings"],
            "user_embeddings": user_embeddings,
            "item_embeddings": item_embeddings,
            "user_outputs": user_outputs,
            "retrieval_outputs": retrieval,
            "contrastive_outputs": contrastive,
            "multimodal_outputs": multimodal_outputs,
            "catalog_semantic_ids": catalog_semantic_ids,
        }

    def generate_recommendations(
        self,
        user_sequences: Dict[str, torch.Tensor],
        catalog_semantic_ids: torch.Tensor,
        top_k: int = 20,
    ):
        self.eval()
        with torch.no_grad():
            rankings = self(
                user_sequences=user_sequences,
                catalog_semantic_ids=catalog_semantic_ids,
                mode="inference",
            )["final_rankings"]
        return [ranking[:top_k] for ranking in rankings]

    def get_user_representation(
        self, user_sequences: Dict[str, torch.Tensor], catalog_semantic_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.encode_user_sequence(user_sequences, catalog_semantic_ids)["user_embeddings"]

    def get_item_representation(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        return self.semantic_item_representations(semantic_ids)

    @staticmethod
    def compute_similarity(
        user_representation: torch.Tensor, item_representation: torch.Tensor
    ) -> torch.Tensor:
        return F.normalize(user_representation, dim=-1) @ F.normalize(
            item_representation, dim=-1
        ).transpose(0, 1)

    def get_model_size(self) -> Dict[str, int]:
        return {
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }

    def save_model(self, save_path: str):
        torch.save({"model_state_dict": self.state_dict()}, save_path)


# Backward-compatible alias for code that imported the old misspelled class.
MASRModel = MGSRModel
