from ..modality import AbstractModality, ModalityConfig
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPModel, CLIPVisionModel
from torchvision.models import resnet50, ResNet50_Weights
import os
import numpy as np
import PIL
import json

CONFIG = json.load(open("config.json"))
    
from typing import Optional


class ImageMoeConfig(ModalityConfig):
    model_type = "image"

    def __init__(
        self,
        modality_name: Optional[str] = None,
        num_patches_per_entry: int = 16,
        max_batch_size: int = 32,
        use_bias_proj: bool = True,
        **kwargs
    ):
        super().__init__(
            modality_name=modality_name,
            num_patches_per_entry=num_patches_per_entry,
            max_batch_size=max_batch_size,
            use_bias_proj=use_bias_proj,
            modality_type=ImageMoeConfig.model_type,
            kwargs=kwargs
        )

class ExpertRouter(nn.Module):
    def __init__(self, num_experts):
        super(ExpertRouter, self).__init__()
        self.num_experts = num_experts
        
        self.model = resnet50(weights='IMAGENET1K_V1')
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )

    def forward(self, x):
        return self.model(x)
    

class ImageModality(AbstractModality):
    config_class = ImageMoeConfig

    def __init__(self, config: ModalityConfig):
        super().__init__(config)

        self.router = ExpertRouter(num_experts=len(CONFIG["image_experts"]))
        self.router.load_state_dict(torch.load(CONFIG["image_router_path"]))
        self.router_image_transform = ResNet50_Weights.IMAGENET1K_V1.transforms()

        self.clip_names = [x['name'] for x in CONFIG["image_experts"]]
        self.image_processors = [CLIPImageProcessor.from_pretrained(x['name']) for x in CONFIG["image_experts"]]
        self.feature_extractors = [CLIPModel.from_pretrained(x['name']) for x in CONFIG["image_experts"]]

    def __call__(self, inputs) -> torch.FloatTensor:
        expert_idx = inputs[-1]
        inputs = inputs[:-1]
        feature_extractor = self.feature_extractors[expert_idx]
        inputs = inputs.to(feature_extractor.device)
        image_features = feature_extractor.get_image_features(pixel_values=inputs)
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
        
        router_image = self.router_image_transform(image)
        router_image = router_image.unsqueeze(0)
        router_output = self.router(router_image)
        expert_idx = torch.argmax(router_output)
        image_processor = self.image_processors[expert_idx]
        processed_image = image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return np.concatenate([processed_image, expert_idx])
    
    @property
    def embedding_size(self) -> int:
        return self.feature_extractor.text_embed_dim
    
    @classmethod
    def from_dict(cls, config_args, **kwargs):
        return ImageMoeConfig.from_dict(config_args, **kwargs)
