from __future__ import annotations

from transformers import PretrainedConfig, PreTrainedModel
from typing import Any, Optional, OrderedDict, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer, PreTrainedModel
from transformers import ProcessorMixin
import json


@dataclass
class ModalityProjectionOutput:
    intermediate_state: torch.FloatTensor
    projection: torch.FloatTensor


class ModalityConfig(PretrainedConfig):
    def __init__(
        self,
        modality_name: Optional[str] = None,
        num_patches_per_entry: int = 16,
        max_batch_size: int = 32,
        use_bias_proj: bool = True,
        modality_type: Optional[str] = None,
        **kwargs
    ):
        self.modality_type = modality_type  # e.g., 'image', 'audio'
        self.modality_name = modality_name  # e.g., 'ClipImage', 'ClipAudio'

        self.num_patches_per_entry = num_patches_per_entry
        self.max_batch_size = max_batch_size
        self.use_bias_proj = use_bias_proj

        super().__init__(**kwargs)

class AbstractModality(ABC, PreTrainedModel):
    def __init__(self, config: ModalityConfig):
        super().__init__(config)

        self.config = config
        self.config_class = ModalityConfig
        # self.feature_extractor = feature_extractor
        self.tokenizer = None

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        ...

    @property
    def num_patches(self) -> int:
        return (self.feature_extractor.image_size // self.feature_extractor.patch_size) ** 2

    @abstractmethod
    def preprocess(self, value: Any, **kwargs) -> Any:
        ...

    def set_requires_grad(self, value: bool):
        for params in self.feature_extractor.parameters():
            params.requires_grad = value



class ModalityWithProjection(nn.Module):
    def __init__(self, modality: AbstractModality, hidden_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = modality.config
        self.modality = modality
        self.hidden_size = hidden_size
        self._dtype = dtype

        self.num_patches_per_entry = modality.config.num_patches_per_entry

        # Define projection layer
        # self.projection = nn.Linear(
        #     modality.embedding_size,
        #     self.hidden_size * self.num_patches_per_entry,
        #     bias=modality.config.use_bias_proj,
        #     dtype=dtype,
        # )
        # Neural network for projection
        self.projection = nn.Sequential(
            nn.Linear(modality.embedding_size, 512, dtype=dtype),
            nn.GELU(),
            nn.Linear(512, 512, dtype=dtype),
            nn.GELU(),
            nn.Linear(512, 512, dtype=dtype),
            nn.GELU(),
            nn.Linear(512, self.hidden_size * self.num_patches_per_entry, dtype=dtype),
        )

    def get_config(self) -> ModalityConfig:
        return self.modality.config

    @property
    def name(self) -> str:
        return self.modality.modality_name

    def freeze_projection_only(self):
        for params in self.projection.parameters():
            params.requires_grad = False
        self.modality.set_requires_grad(True)

    def freeze_modality_only(self):
        for params in self.projection.parameters():
            params.requires_grad = True
        self.modality.set_requires_grad(False)

    def freeze_all(self):
        for params in self.parameters():
            params.requires_grad = False

    def unfreeze_all(self):
        for params in self.parameters():
            params.requires_grad = True

    def preprocess(self, value: Any, **kwargs) -> Any:
        return self.modality.preprocess(value, **kwargs)

    def forward(self, value: Any) -> ModalityProjectionOutput:
        hidden_state = self.modality(value).to(self._dtype)

        # Flatten and project
        hidden_state = torch.flatten(hidden_state, start_dim=1)
        projection = self.projection(hidden_state)

        # Reshape projection
        projection = projection.view(-1,
                                     self.num_patches_per_entry, self.hidden_size)

        return ModalityProjectionOutput(
            intermediate_state=hidden_state,
            projection=projection,
        )
