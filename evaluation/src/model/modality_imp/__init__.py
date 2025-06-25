from .image_modality import ImageModality, ImageConfig
from .three_d_modality import CTModality, CTConfig
from transformers import AutoConfig, AutoModel

MODALITY_FROM_NAME = {
    "ImageModality": ImageModality,
    "3DModality": CTModality,
}
MODALITY_CONFIG_FROM_NAME = {
    "ImageModality" : ImageConfig,
    "3DModality": CTConfig,
}

for modality_name in MODALITY_FROM_NAME.keys():
    cls = MODALITY_FROM_NAME[modality_name]
    config_cls = MODALITY_CONFIG_FROM_NAME[modality_name]
    AutoConfig.register(getattr(config_cls, "model_type"), cls)
    AutoModel.register(config_cls, cls)