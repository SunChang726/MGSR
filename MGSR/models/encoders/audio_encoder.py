from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


class AudioEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        hidden_dim: int = 768,
        output_dim: int = 256,
        sample_rate: int = 16000,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.freeze_backbone = freeze_backbone
        self.processor = None
        self.wav2vec2 = None
        self.projection = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        self.precomputed_projection = nn.Sequential(nn.LazyLinear(output_dim), nn.LayerNorm(output_dim))

    def _load_backbone(self):
        if self.wav2vec2 is not None:
            return
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
        except ImportError as error:
            raise ImportError("transformers is required to encode raw audio") from error
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(self.model_name)
        if self.freeze_backbone:
            self.wav2vec2.requires_grad_(False)
        self.wav2vec2.to(self.projection[0].weight.device)

    def forward(self, audio_inputs: Union[torch.Tensor, List[np.ndarray], List[str]]):
        if torch.is_tensor(audio_inputs) and audio_inputs.dim() == 2 and audio_inputs.size(-1) == self.hidden_dim:
            return self.encode_synthetic_features(audio_inputs)
        self._load_backbone()
        waveforms = self._prepare_waveforms(audio_inputs).to(self.projection[0].weight.device)
        hidden = self.wav2vec2(waveforms).last_hidden_state.mean(dim=1)
        return self.projection(hidden)

    def _prepare_waveforms(self, audio_inputs):
        if torch.is_tensor(audio_inputs):
            return audio_inputs.float()
        arrays = []
        for value in audio_inputs:
            if isinstance(value, str):
                try:
                    import librosa
                except ImportError as error:
                    raise ImportError("librosa is required to load audio files") from error
                value, _ = librosa.load(value, sr=self.sample_rate)
            arrays.append(torch.as_tensor(value, dtype=torch.float))
        max_length = max(array.numel() for array in arrays)
        return torch.stack([torch.nn.functional.pad(array, (0, max_length - array.numel())) for array in arrays])

    def encode_synthetic_features(self, features: torch.Tensor):
        return self.precomputed_projection(features.float())


class SpectrogramAudioEncoder(nn.Module):

    def __init__(self, output_dim: int = 256, **_):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, spectrograms: torch.Tensor):
        if spectrograms.dim() == 3:
            spectrograms = spectrograms.unsqueeze(1)
        return self.encoder(spectrograms.float())


class MultiModalAudioEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, use_wav2vec: bool = True, use_spectrogram: bool = True):
        super().__init__()
        self.use_wav2vec = use_wav2vec
        self.use_spectrogram = use_spectrogram
        self.wav2vec_encoder = AudioEncoder(output_dim=output_dim) if use_wav2vec else None
        self.spectrogram_encoder = SpectrogramAudioEncoder(output_dim=output_dim) if use_spectrogram else None
        count = int(use_wav2vec) + int(use_spectrogram)
        self.fusion = nn.Linear(output_dim * count, output_dim) if count > 1 else nn.Identity()

    def forward(self, audio_inputs, spectrograms=None):
        features = []
        if self.wav2vec_encoder is not None:
            features.append(self.wav2vec_encoder(audio_inputs))
        if self.spectrogram_encoder is not None:
            if spectrograms is None:
                raise ValueError("spectrograms are required when use_spectrogram=True")
            features.append(self.spectrogram_encoder(spectrograms))
        return self.fusion(torch.cat(features, dim=-1) if len(features) > 1 else features[0])
