from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        hidden_dim: int = 768,
        output_dim: int = 256,
        image_size: int = 224,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.image_size = image_size
        self.freeze_backbone = freeze_backbone
        self.vit = None
        self.projection = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        self.precomputed_projection = nn.Sequential(nn.LazyLinear(output_dim), nn.LayerNorm(output_dim))

    def _load_backbone(self):
        if self.vit is not None:
            return
        try:
            from transformers import ViTModel
        except ImportError as error:
            raise ImportError("transformers is required to encode raw images") from error
        self.vit = ViTModel.from_pretrained(self.model_name)
        if self.freeze_backbone:
            self.vit.requires_grad_(False)
        self.vit.to(self.projection[0].weight.device)

    def forward(self, images) -> torch.Tensor:
        if torch.is_tensor(images) and images.dim() == 2:
            return self.encode_synthetic_features(images)
        self._load_backbone()
        pixels = self._preprocess_images(images) if isinstance(images, list) else images
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        pixels = pixels.to(self.projection[0].weight.device)
        pooled = self.vit(pixel_values=pixels).last_hidden_state[:, 0]
        return self.projection(pooled)

    def _preprocess_images(self, images: List[Union[np.ndarray, object]]) -> torch.Tensor:
        try:
            from PIL import Image
            from torchvision import transforms
        except ImportError as error:
            raise ImportError("Pillow and torchvision are required to preprocess raw images") from error
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        processed = []
        for image in images:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            processed.append(transform(image.convert("RGB")))
        return torch.stack(processed)

    def encode_synthetic_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.precomputed_projection(features.float())


class CNNVisionEncoder(nn.Module):

    def __init__(self, output_dim: int = 256, pretrained: bool = True, **_):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as error:
            raise ImportError("torchvision is required for CNNVisionEncoder") from error
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        input_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, images: torch.Tensor):
        return self.projection(self.backbone(images))


class MultiScaleVisionEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, scales=(224, 384, 512)):
        super().__init__()
        self.encoders = nn.ModuleList(
            [VisionEncoder(output_dim=output_dim, image_size=scale) for scale in scales]
        )
        self.fusion = nn.Linear(output_dim * len(scales), output_dim)

    def forward(self, images):
        return self.fusion(torch.cat([encoder(images) for encoder in self.encoders], dim=-1))
