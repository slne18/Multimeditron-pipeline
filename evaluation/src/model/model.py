import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple, Any, Dict
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from .modality import ModalityWithProjection, ModalityConfig
from ..utils import expand_attachments, pad_sequence, pad_tensor_sequence, batch
from .modality_imp import * # In order to register the modality classes
import logging
import os
import json
import tempfile

logger = logging.getLogger(__name__)

# try:
#     import deepspeed
# except ImportError:
#     logger.warning(
#         "Deepspeed is not installed, running without it. You must not run this model with deepspeed."
#         "If you do so, issues will arise."
#     )

try:
    from torch.distributed.fsdp import \
        FullyShardedDataParallel as FSDP, \
        CPUOffload

except ImportError:
    logger.warning(
        "Fully Sharded Data Parallel is not installed, running without it. You must not run this model with FSDP."
        "If you do so, issues will arise."
    )



@dataclass
class MultimodalRawInput:
    type: str
    value: Any
    preprocessor_kwargs: Dict[str, Any] = None

@dataclass
class MultimodalProcessedInput:
    type: str
    value: torch.Tensor

@dataclass
class MultimodalConfig(PretrainedConfig):
    model_type = "multimodal"

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        modalities: List[ModalityConfig] = [],
        attachment_token_idx: int = 1,
        pad_token_idx: int = 0,
        eos_token_idx: int = 0,
        padding_side: str = "left",
        initializer_range: float = 0.02,
        llm_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.modalities = modalities
        self.attachment_token_idx = attachment_token_idx
        self.pad_token_idx = pad_token_idx
        self.eos_token_idx = eos_token_idx
        self.padding_side = padding_side
        self.initializer_range = initializer_range
        self.llm_path = llm_path

    def to_dict(self):
        output = super().to_dict()
        output['modalities'] = [modality_config.to_dict()
                                for modality_config in self.modalities]
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        modalities_dict_list = config_dict.pop('modalities', [])
        
        modalities = []
        for modality_dict in modalities_dict_list:
            # Hacky stuff
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
                json.dump(modality_dict, tmp_file, ensure_ascii=False, indent=2)
            modalities.append(AutoConfig.from_pretrained(tmp_file.name))
        
        if kwargs["return_unused_kwargs"]:
            config, kwargs = super().from_dict(config_dict, **kwargs)
            config.modalities = modalities
            return config, kwargs
        
        config = super().from_dict(config_dict, kwargs)
        config.modalities = modalities
        return config


class MultiModalModelForCausalLM(PreTrainedModel):
    config_class = MultimodalConfig
    base_model_prefix = "model"

    def __init__(
        self,
        config: MultimodalConfig,
        dtype=torch.bfloat16,
        use_fsdp=False,
        fsdp_cpu_offload=False,
    ):
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_pretrained(config.llm_path, torch_dtype=dtype)
        self.model.resize_token_embeddings(config.vocab_size)

        # Add the language model to the transformer
        self.modalities_by_type = {}
        self.modalities = nn.ModuleList()

        for modality_config in config.modalities:
            # Retrieve the modality and the number of patches per entry
            modality = AutoModel.from_config(modality_config)

            # Ensure there is a single modality per type
            if modality_config.modality_type in self.modalities_by_type:
                raise ValueError(
                    f"Modality type {modality_config.modality_type} has already been registered"
                )

            self.modalities_by_type[modality_config.modality_type] = len(
                self.modalities)
            self.modalities.append(
                ModalityWithProjection(
                    modality,
                    hidden_size=self.model.config.hidden_size,
                    dtype=dtype,
                )
            )

        # Make use of FSDP if enabled
        if not "FSDP" in globals() and use_fsdp:
            raise ImportError("FSDP is not installed, cannot use it.")

        if fsdp_cpu_offload and not use_fsdp:
            raise ValueError(
                "Cannot use CPU offload without FSDP, please enable FSDP."
            )

        if use_fsdp:
            logger.info("Using fully sharded data parallelism (pytorch-fsdp)")

            self.model = FSDP(
                self.model,
                cpu_offload=CPUOffload(offload_params=fsdp_cpu_offload)
            )

        # Post init
        self.post_init()

        # Computation of the model may vary accross different passes, this is however
        # not supported by ZeRO-3, therefore we need to disable the gradient checkpointing
        # for any module earlier than the transformer so that z3 don't register hooks on them
        # if "deepspeed" in globals():
        #     logger.info("Disabling ZeRO-3 gradient checkpointing for all modalities")
        #     deepspeed.utils.set_z3_leaf_modules(self, [ModalityWithProjection])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def freeze_for_alignment(self):
        for modality in self.modalities:
            modality.freeze_modality_only()
        for params in self.model.parameters():
            params.requires_grad = False

    def freeze_for_lm(self):
        for modality in self.modalities:
            modality.freeze_all()
        for params in self.model.parameters():
            params.requires_grad = True

    def unfreeze(self):
        for modality in self.modalities:
            modality.unfreeze_all()
        for params in self.model.parameters():
            params.requires_grad = True

    def processors(self):
        return {modality.get_config().modality_type: modality for modality in self.modalities}

    def get_model(self):
        return self.model

    def _get_modality_by_name(self, name: str) -> ModalityWithProjection:
        if name not in self.modalities_by_type:
            raise KeyError(
                f"No modality registered in the model that can handle modality named: {name}"
            )
        index = self.modalities_by_type[name]
        modality = self.modalities[index]
        if not isinstance(modality, ModalityWithProjection):
            raise TypeError(
                f"Registered modality {name} is not of type ModalityWithProjection")
        return modality

    def preprocess_modalities(self, multimodal_unprocessed: List[List[MultimodalRawInput]]) -> List[List[MultimodalProcessedInput]]:
        xs = []
        for single_batch in multimodal_unprocessed:
            current_batch_xs = []

            for sample in single_batch:
                modality = self._get_modality_by_name(sample.type)
                if sample.preprocessor_kwargs is None:
                    output_value = modality.preprocess(sample.value)
                else:
                    output_value = modality.preprocess(sample.value, **sample.preprocessor_kwargs)

                current_batch_xs.append(MultimodalProcessedInput(
                    type=sample.type,
                    value=output_value
                ))

            xs.append(current_batch_xs)
        return xs

    def encode_modalities(self, processed_modalities: List[List[MultimodalProcessedInput]]) -> List[List[torch.FloatTensor]]:
        # Batch processing of modalities
        grouped_modalities = {key: [] for key in self.modalities_by_type.keys()}

        for i, single_batch in enumerate(processed_modalities):
            for j, sample in enumerate(single_batch):
                grouped_modalities[sample.type].append((i, j, sample.value))

        # Group each embedding together
        # and compute the projection
        output_embeddings = [[None] * len(single_batch) for single_batch in processed_modalities]

        for modality_name, samples in grouped_modalities.items():
            modality = self._get_modality_by_name(modality_name)
            modality_config = modality.get_config()

            # Apply each modality in batch
            for batched_samples in batch(samples, modality_config.max_batch_size):
                ix, jx, samples = zip(*batched_samples)
                batched_samples = torch.stack(samples, dim=0)

                # Compute the projection
                output_value = modality(batched_samples)

                # Unpack the output
                for i, j, output in zip(ix, jx, output_value.projection):
                    output_embeddings[i][j] = output

        # Finally, return the embeddings
        return output_embeddings

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    def encode_modality_inputs(
        self,
        input_ids,
        modalities: List[List[MultimodalRawInput]],
    ) -> torch.FloatTensor:

        # Preprocess modalities
        processed_modalities = self.preprocess_modalities(modalities)

        # Compute the embedding for each modality
        encoded_modality_features = self.encode_modalities(processed_modalities)
        
        # Perform the embedding expansion
        cpu_input_ids = input_ids.clone().cpu()
        embeddings = self.model.get_input_embeddings()(input_ids)

        # Replace all the <|attachment|> tokens with the modality embeddings
        for batch_idx in range(cpu_input_ids.shape[0]):
            # If no attachment points are present, skip
            if len(encoded_modality_features[batch_idx]) == 0:
                continue

            # Iterate over each attachment point
            i = 0
            j = 0
            while i < cpu_input_ids.shape[1]:
                if cpu_input_ids[batch_idx, i] == self.config.attachment_token_idx:
                    # Better error handling for debugging dumb mistakes
                    if j >= len(encoded_modality_features[batch_idx]):
                        raise ValueError(
                            "Number of attachment points in the input is greater than the number of modality embeddings"
                            f"(batch_idx={batch_idx}, i={i}, j={j}, len={len(encoded_modality_features[batch_idx])})"
                        )

                    # Replace the token with the modality embedding
                    modality_embedding = encoded_modality_features[batch_idx][j]
                    num_tokens = modality_embedding.shape[0]
                    embeddings[batch_idx, i:i + num_tokens] = modality_embedding
                    i += num_tokens
                    j += 1

                else:
                    i += 1
        
        # Return the embeddings
        return embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        multimodal_inputs = None,
        return_dict: Optional[bool] = True,
        cache_position=None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if multimodal_inputs is None:
            multimodal_inputs = [[]] * input_ids.shape[0]
        if inputs_embeds is None:
            inputs_embeds = self.encode_modality_inputs(input_ids, multimodal_inputs)

        # Run the transformer model
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )

    def generate(
        self,
        input_ids: torch.LongTensor = None,
        multimodal_inputs = None,
        max_length=512,
        temperature=0.1,
        do_sample=True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if multimodal_inputs is None:
            multimodal_inputs = [[]] * input_ids.shape[0]
        
        # Convert modalities to MultimodalRawInput if necessary
        raw_inputs = []
        for sample in multimodal_inputs:
            sample_raw_inputs = []
            for modality in sample:
                if isinstance(modality, MultimodalRawInput):
                    sample_raw_inputs.append(modality)
                else:
                    sample_raw_inputs.append(MultimodalRawInput(type=modality["type"], value=modality["value"]))
            raw_inputs.append(sample_raw_inputs)
                    
        inputs_embeds = self.encode_modality_inputs(input_ids, raw_inputs)

        # Run the transformer model
        generated_tokens = []
        past_key_values = None
        next_token_embedding = inputs_embeds
        finished_mask = torch.zeros(input_ids.shape[0])
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass with embeddings
                outputs = self.model(inputs_embeds=next_token_embedding, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

                # Get the logits and apply softmax to get token probabilities
                logits = outputs.logits[:, -1, :].squeeze(1)  # Get logits for the last token
                logits = logits / temperature
                softmax = F.softmax(logits, dim=-1)

                if do_sample:
                    next_token_id = []
                    for sample_softmax in softmax:
                        next_token_id.append(torch.multinomial(sample_softmax, num_samples=1))
                    next_token_id = torch.cat(next_token_id).unsqueeze(0).cpu()
                else:
                    next_token_id = torch.argmax(softmax, dim=-1).unsqueeze(0).cpu()

                for i in range(next_token_id.shape[1]):
                    if finished_mask[i]:
                        next_token_id[0, i] = self.config.eos_token_idx

                # Append the next token to your sequence
                generated_tokens.append(next_token_id)

                finished_mask = torch.logical_or(finished_mask, next_token_id.flatten() == self.config.eos_token_idx)
                if torch.all(finished_mask):
                    break

                # Update the input_embeddings with the embedding of the newly generated token
                next_token_embedding = self.model.get_input_embeddings()(next_token_id.to(input_ids.device)).transpose(1, 0)
        
        return torch.cat(generated_tokens).transpose(1, 0)
