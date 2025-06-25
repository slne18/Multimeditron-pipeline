from ..modality import AbstractModality, ModalityConfig
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPModel, CLIPVisionModel
import os
import numpy as np
import PIL
    
from typing import Optional


class ImageConfig(ModalityConfig):
    model_type = "image"

    def __init__(
        self,
        modality_name: Optional[str] = None,
        num_patches_per_entry: int = 16,
        max_batch_size: int = 64,
        use_bias_proj: bool = True,
        **kwargs
    ):
        super().__init__(
            modality_name=modality_name,
            num_patches_per_entry=num_patches_per_entry,
            max_batch_size=max_batch_size,
            use_bias_proj=use_bias_proj,
            modality_type=ImageConfig.model_type,
            kwargs=kwargs
        )


class ImageModality(AbstractModality):
    config_class = ImageConfig

    # clip_name="openai/clip-vit-large-patch14"):
    def __init__(self, config: ModalityConfig):
        super().__init__(config)

        clip_name = getattr(config, "clip_name",
                            "openai/clip-vit-large-patch14")
        self.clip_name = clip_name
        self.vision_tower_name = clip_name
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name)
        self.feature_extractor = CLIPModel.from_pretrained(self.vision_tower_name)

    def __call__(self, inputs) -> torch.FloatTensor:
        inputs = inputs.to(self.feature_extractor.device)
        image_features = self.feature_extractor.get_image_features(pixel_values=inputs)
        return image_features

    def preprocess(self, image, base_path = ""):
        if isinstance(image, str):
            image_path = os.path.join(base_path, image)
            # Load and preprocess the image (expecting .npy format for 3D medical images)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file {image_path} not found")
            # Load png/jpg/jpeg images
            image = PIL.Image.open(image_path).convert("RGB")
            # to torch tensor
            image = np.array(image)
            # image = np.resize(image, (1, 3, 224, 224))
        elif isinstance(image, np.ndarray):
            image = image
            
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
    
    @property
    def embedding_size(self) -> int:
        return self.feature_extractor.text_embed_dim
    
    @classmethod
    def from_dict(cls, config_args, **kwargs):
        return ImageConfig.from_dict(config_args, **kwargs)
