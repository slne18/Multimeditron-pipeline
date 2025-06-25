from torch.utils.data import Dataset, IterableDataset
from typing import Dict, List, Any, Optional, Union, Iterator
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from dataclasses import dataclass
from src.model.modality import ModalityWithProjection
from src.model.model import MultimodalRawInput
from src.utils import redistribute_batches
from src.model.prompt_tokenizers import Llama3PromptTokenizer
import json
import random
import itertools
import torch

IGNORE_TOKEN_INDEX = -100  # This value is hardcoded in the transformers library

class NonTensorBatchItem:
    def __init__(self, data):
        self.data = data
        
class MultimodalDataset(Dataset):
    def __init__(
        self,
        dataset: Iterator | IterableDataset | str,
        tokenizer: PreTrainedTokenizerBase,
        modality_processors: Dict[str, ModalityWithProjection],
        attachment_token_idx: int,
        sample_count_per_block: int = 1024,
        context_length: int = 1024,
        shuffle: bool = False,
        packing: bool = False,
    ):
        if isinstance(dataset, str):
            def loader_fn():
                with open(dataset, 'r') as f:
                    for line in f:
                        yield json.loads(line)
            self.dataset = loader_fn()

        # Store the dataset and tokenizer
        self.tokenizer = tokenizer
        self.modality_processors = modality_processors
        self.sample_count_per_block = sample_count_per_block
        self.context_length = context_length
        self.shuffle = shuffle
        self.packing = packing
        
        modalities_num_embeddings = {mod_name: mod.num_patches_per_entry for mod_name, mod in modality_processors.items()}
        self.prompt_tokenizer = Llama3PromptTokenizer(
            tokenizer=self.tokenizer,
            modalities_num_embeddings=modalities_num_embeddings,
            attachment_token_idx=attachment_token_idx
        )

    def _apply_packing(self, block):
        input_ids = []
        labels = []
        attention_mask = []
        position_ids = []
        modalities = []

        # Compute the total number of tokens in the block
        total_tokens = sum(len(entry["input_ids"]) for entry in block)

        # Pack the dataset
        packs: List[List[Dict[str, torch.Tensor]]] = redistribute_batches(block, (total_tokens // self.context_length) + 1)
        
        # Process each pack
        for pack in packs:
            input_ids.append(torch.cat([entry["input_ids"] for entry in pack], dim=1))
            labels.append(torch.cat([entry["labels"] for entry in pack], dim=1))
            attention_mask.append(torch.cat([entry["attention_mask"] for entry in pack], dim=1))
            position_ids.append(torch.cat([entry["position_ids"] for entry in pack], dim=1))
            modalities.append([entry["modalities"] for entry in pack])

        raise NotImplementedError("This is a placeholder implementation.")
            
        return [{
            "input_ids": input_ids[i],
            "labels": labels[i],
            "attention_mask": attention_mask[i],
            "position_ids": position_ids[i],
            "modalities": modalities[i]
        } for i in range(len(packs))]

    def _process_block(self, block):
        # If packing is enabled, we pack the block
        if self.packing:
            block = self._apply_packing(block)
        
        # Shuffle the block if needed
        if self.shuffle:
            random.shuffle(block)
            
        # Return the block
        return block

    def __iter__(self):
        while True:
            # First we load a single block of samples
            block = []
            for entry in itertools.islice(self.dataset, self.sample_count_per_block):
                block.append(entry)
                
            # If the block is empty, we have reached the end of the dataset
            if len(block) == 0:
                break

            # Process the block
            # This is where we load external modalities, 
            # apply the tokenizers, perform packing, etc.
            processed_block = self._process_block(block)
            
            # Yield the processed block
            yield from processed_block
            

@dataclass
class DataCollatorForMultimodal(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    modality_processors: Dict[str, ModalityWithProjection]
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
            Args:
            
            features (List[Dict[str, Any]]): List of batches, each Dictionary contains the following keys: 
                - input_ids (List[int]): List of input token ids.
                - labels (List[int]): List of label token ids.
                - modalities (List[Dict[str, Any]]): List of modalities, each Dictionary contains the following keys:
                    - type (str): Modality type.
                    - value (Any): Modality value.
                Each element in the list is a sample.
        """
        # Separate features by modality
        batch = {}

        feature_names = ['input_ids', 'labels']
        
        text_features = {
            'input_ids' : [],
            'labels' : [],
        }
        
        for sample in features:
            for name in feature_names:
                text_features[name].append(sample[name])

        # Use the tokenizer's pad method to handle padding for text features
        input_batch = {key: text_features[key] for key in ["input_ids"]}
        text_batch = self.tokenizer.pad(
            input_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Little hack to pad the labels (trust me)
        label_batch = self.tokenizer.pad(
            {"input_ids" : text_features["labels"]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        text_batch["labels"] = label_batch["input_ids"]

        batch.update(text_batch)

        # Process modalities
        batch_size = len(features)

        multimodal_inputs = []
        for i in range(batch_size):
            sample = features[i]
            multimodal_sample = []
            
            for modality in sample["modalities"]:
                multimodal_sample.append(
                    MultimodalRawInput(
                        type=modality["type"],
                        value=modality["value"],
                        preprocessor_kwargs=modality.get("preprocessor_kwargs", None)
                    )
                )

            multimodal_inputs.append(multimodal_sample)
        batch['multimodal_inputs'] = NonTensorBatchItem(multimodal_inputs)
        
        return batch

    def tf_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "TensorFlow is not supported for multimodal data collation.")

    def numpy_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "NumPy is not supported for multimodal data collation.")
