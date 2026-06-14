from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class TextEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        output_dim: int = 256,
        max_length: int = 256,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.freeze_bert = freeze_bert
        self.tokenizer = None
        self.bert = None
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.precomputed_projection = nn.Sequential(
            nn.LazyLinear(output_dim),
            nn.LayerNorm(output_dim),
        )

    def _load_backbone(self):
        if self.bert is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError("transformers is required to encode raw text") from error
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        if self.freeze_bert:
            self.bert.requires_grad_(False)
        self.bert.to(self.projection[0].weight.device)

    def forward(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(texts):
            return self.encode_synthetic_features(texts)
        if isinstance(texts, str):
            texts = [texts]
        self._load_backbone()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = self.projection[0].weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = self.bert(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(pooled)

    def encode_synthetic_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.precomputed_projection(features.float())

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        return torch.cat([self(texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)])


class MultilingualTextEncoder(TextEncoder):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_name", "distilbert-base-multilingual-cased")
        super().__init__(**kwargs)


class DomainAdaptiveTextEncoder(TextEncoder):
    def __init__(self, domain_vocab: Optional[Dict[str, int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.domain_vocab = domain_vocab or {}
        self.domain_embedding = (
            nn.Embedding(len(self.domain_vocab), self.output_dim) if self.domain_vocab else None
        )

    def forward(self, texts, domains: Optional[List[str]] = None):
        embeddings = super().forward(texts)
        if self.domain_embedding is not None and domains is not None:
            domain_ids = torch.tensor(
                [self.domain_vocab.get(domain, 0) for domain in domains],
                device=embeddings.device,
            )
            embeddings = embeddings + self.domain_embedding(domain_ids)
        return embeddings
