from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embeddings = torch.empty(num_embeddings, embedding_dim)
        nn.init.uniform_(embeddings, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer("embedding", embeddings)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embed_avg", embeddings.clone())

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = inputs.shape
        flat = inputs.reshape(-1, self.embedding_dim)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + self.embedding.square().sum(dim=1)
            - 2 * flat @ self.embedding.transpose(0, 1)
        )
        indices = distances.argmin(dim=1)
        one_hot = F.one_hot(indices, self.num_embeddings).to(flat.dtype)
        quantized = F.embedding(indices, self.embedding).view(shape)

        if self.training:
            with torch.no_grad():
                counts = one_hot.sum(dim=0)
                sums = one_hot.transpose(0, 1) @ flat
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(sums, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                normalized = (
                    (self.cluster_size + self.epsilon)
                    / (total + self.num_embeddings * self.epsilon)
                    * total
                )
                self.embedding.copy_(self.embed_avg / normalized.unsqueeze(1).clamp_min(self.epsilon))

        codebook_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, indices.view(shape[:-1])


class HierarchicalQuantizer(nn.Module):

    def __init__(
        self,
        input_dim: int = 256,
        coarse_codebook_size: int = 256,
        fine_codebook_size: int = 1024,
        coarse_dim: int = 256,
        fine_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.coarse_codebook_size = coarse_codebook_size
        self.fine_codebook_size = fine_codebook_size
        self.coarse_dim = coarse_dim
        self.fine_dim = fine_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = 1e-5

        self.coarse_projection = nn.Linear(input_dim, coarse_dim)
        self.fine_projection = nn.Linear(input_dim, fine_dim)
        self.coarse_quantizer = VectorQuantizer(
            coarse_codebook_size, coarse_dim, commitment_cost, decay
        )

        fine_codebooks = torch.empty(coarse_codebook_size, fine_codebook_size, fine_dim)
        nn.init.uniform_(fine_codebooks, -1.0 / fine_codebook_size, 1.0 / fine_codebook_size)
        self.register_buffer("fine_codebooks", fine_codebooks)
        self.register_buffer(
            "fine_cluster_size", torch.ones(coarse_codebook_size, fine_codebook_size)
        )
        self.register_buffer("fine_embed_avg", fine_codebooks.clone())
        self.reconstruction_projection = nn.Linear(coarse_dim + fine_dim, input_dim)

    def _quantize_fine(
        self, fine_features: torch.Tensor, coarse_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = fine_features.shape
        flat = fine_features.reshape(-1, self.fine_dim)
        flat_coarse = coarse_indices.reshape(-1)
        codebooks = self.fine_codebooks[flat_coarse]
        distances = (flat.unsqueeze(1) - codebooks).square().sum(dim=-1)
        fine_indices = distances.argmin(dim=-1)
        row = torch.arange(flat.size(0), device=flat.device)
        quantized = codebooks[row, fine_indices].view(shape)

        if self.training:
            with torch.no_grad():
                pair_indices = flat_coarse * self.fine_codebook_size + fine_indices
                one_hot = F.one_hot(
                    pair_indices, self.coarse_codebook_size * self.fine_codebook_size
                ).to(flat.dtype)
                counts = one_hot.sum(0).view_as(self.fine_cluster_size)
                sums = (one_hot.transpose(0, 1) @ flat).view_as(self.fine_embed_avg)
                self.fine_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.fine_embed_avg.mul_(self.decay).add_(sums, alpha=1 - self.decay)
                self.fine_codebooks.copy_(
                    self.fine_embed_avg
                    / self.fine_cluster_size.unsqueeze(-1).clamp_min(self.epsilon)
                )

        codebook_loss = F.mse_loss(quantized, fine_features.detach())
        commitment_loss = F.mse_loss(quantized.detach(), fine_features)
        loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = fine_features + (quantized - fine_features).detach()
        return quantized, loss, fine_indices.view(shape[:-1])

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        coarse_features = self.coarse_projection(inputs)
        fine_features = self.fine_projection(inputs)
        coarse_quantized, coarse_loss, coarse_indices = self.coarse_quantizer(coarse_features)
        fine_quantized, fine_loss, fine_indices = self._quantize_fine(
            fine_features, coarse_indices
        )
        reconstructed = self.reconstruction_projection(
            torch.cat([coarse_quantized, fine_quantized], dim=-1)
        )
        return {
            "quantized": reconstructed,
            "coarse_quantized": coarse_quantized,
            "fine_quantized": fine_quantized,
            "coarse_indices": coarse_indices,
            "fine_indices": fine_indices,
            "loss": coarse_loss + fine_loss,
            "coarse_loss": coarse_loss,
            "fine_loss": fine_loss,
        }

    def encode(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        results = self.forward(inputs)
        return results["coarse_indices"], results["fine_indices"]

    def decode(self, coarse_indices: torch.Tensor, fine_indices: torch.Tensor) -> torch.Tensor:
        coarse = F.embedding(coarse_indices, self.coarse_quantizer.embedding)
        fine = self.fine_codebooks[coarse_indices, fine_indices]
        return self.reconstruction_projection(torch.cat([coarse, fine], dim=-1))


class SemanticIDGenerator(nn.Module):

    def __init__(
        self,
        input_dim: int = 256,
        coarse_vocab_size: int = 256,
        fine_vocab_size: int = 1024,
        num_transformer_layers: int = 1,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_attention_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.semantic_transform = nn.TransformerEncoder(layer, num_transformer_layers)
        self.hierarchical_quantizer = HierarchicalQuantizer(
            input_dim=input_dim,
            coarse_codebook_size=coarse_vocab_size,
            fine_codebook_size=fine_vocab_size,
            coarse_dim=input_dim,
            fine_dim=input_dim,
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        squeeze = features.dim() == 2
        sequence = features.unsqueeze(1) if squeeze else features
        transformed = self.semantic_transform(sequence)
        results = self.hierarchical_quantizer(transformed)
        semantic_ids = torch.stack(
            [results["coarse_indices"], results["fine_indices"]], dim=-1
        )
        output = {
            "semantic_ids": semantic_ids,
            "coarse_ids": results["coarse_indices"],
            "fine_ids": results["fine_indices"],
            "transformed": transformed,
            **results,
        }
        if squeeze:
            for key in ("semantic_ids", "coarse_ids", "fine_ids", "transformed", "quantized"):
                output[key] = output[key].squeeze(1)
        return output

    def generate_semantic_ids(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(features)["semantic_ids"]


class ProductQuantizer(nn.Module):

    def __init__(self, input_dim: int = 256, num_subspaces: int = 8, codebook_size: int = 256):
        super().__init__()
        if input_dim % num_subspaces != 0:
            raise ValueError("input_dim must be divisible by num_subspaces")
        self.num_subspaces = num_subspaces
        self.subspace_dim = input_dim // num_subspaces
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(codebook_size, self.subspace_dim) for _ in range(num_subspaces)]
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        parts = inputs.view(*inputs.shape[:-1], self.num_subspaces, self.subspace_dim)
        outputs = [quantizer(parts[..., index, :]) for index, quantizer in enumerate(self.quantizers)]
        return {
            "quantized": torch.cat([output[0] for output in outputs], dim=-1),
            "indices": torch.stack([output[2] for output in outputs], dim=-1),
            "loss": torch.stack([output[1] for output in outputs]).sum(),
        }


AdaptiveQuantizer = VectorQuantizer
