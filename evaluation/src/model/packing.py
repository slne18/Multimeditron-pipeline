from torch.utils.data import IterableDataset
from src.model.modality import ModalityWithProjection
from typing import Generator, List, Dict, Any
from transformers import PreTrainedTokenizerBase
from src.model.prompt_tokenizers import Llama3PromptTokenizer
from src.utils import redistribute_batches
import torch
from datasets import load_dataset
import os
from huggingface_hub import repo_exists
import json


class JSONLGenerator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file = open(file_path, 'r')

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if not line:
            raise StopIteration
        return eval(line)

    def __del__(self):
        self.file.close()
        
    def reset(self):
        self.file = open(self.file_path, 'r')

class PackedDataset(IterableDataset):
    def __init__(self, stream: Generator|str, ctx_len: int, 
                 separator_token_idx: int,
                 tokenizer: PreTrainedTokenizerBase,
                 modality_processors: Dict[str, ModalityWithProjection],
                 attachment_token_idx: int, ignore_index: int = -100, lookahead: int = 1, 
                 infinite: bool = False, separate: bool = False, shuffle: bool = False,
                 preprocessor_kwargs: Dict[str, Any] = None):
        """
        Args:
            stream (Generator): A generator that yields samples from the dataset.
            ctx_len (int): The maximum context length. i.e the maximum number of tokens in a single pack.
            separator_token_idx (int): Index of the separator token.
            ignore_index (int): Index to ignore in the loss computation.
            lookahead (int): Number of packs to prepare in advance.
            infinite (bool): Whether to loop the stream infinitely.
            separate (bool): Whether to separate the sequences with the separator token.
            preprocessor_kwargs (Dict[str, Any]): Additional arguments to pass to the processor during the preprocessing part
        """
        
        # If the stream is a string, check if it is a HuggingFace dataset if not, open the file as a stream
        if isinstance(stream, str):
            if len(stream.split("/")) == 2 and repo_exists(stream):
                dataset = load_dataset(stream, streaming=True)
                stream = dataset['train']
            elif os.path.isfile(stream):
                # Let's generate a Generator from the jsonl file
                stream = JSONLGenerator(stream)
            else:
                raise ValueError(f"stream has value {stream}. The stream must be a valid HuggingFace dataset or a file path.")
            
        self.stream = stream
        self.ctx_len = ctx_len
        self.separator_token_idx = separator_token_idx
        self.ignore_index = ignore_index
        self.lookahead = lookahead 
        self.infinite = infinite
        self.separate = separate
        self.shuffle = shuffle
        self.preprocessor_kwargs = preprocessor_kwargs
        modalities_num_embeddings = {mod_name: mod.num_patches_per_entry for mod_name, mod in modality_processors.items()}

        self.prompt_tokenizer = Llama3PromptTokenizer(
            tokenizer=tokenizer,
            modalities_num_embeddings=modalities_num_embeddings,
            attachment_token_idx=attachment_token_idx
        )
    
    def process_sample(self, sample):
        processed_sample = {}

        # Process text input
        if 'text' in sample:
            text = sample['text']
            
            tokenized = self.prompt_tokenizer.tokenize_text(
                prompt=text,
                modalities=sample["modalities"]
            )
        elif 'conversations' in sample:
            # Process conversation data
            conversation = sample['conversations']
            
            tokenized = self.prompt_tokenizer.tokenize_conversation(
                prompt=conversation,
                modalities=sample["modalities"]
            )
        else:
            raise ValueError(
                "Each sample must contain either 'text' or 'conversations'.")

        processed_sample['input_ids'] = tokenized['input_ids'].squeeze(0)
        processed_sample['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        processed_sample['labels'] = tokenized['labels'].squeeze(0)

        # Add additional kwargs to modalities
        processed_sample["modalities"] = []
        for modality in sample["modalities"]:
            modality["preprocessor_kwargs"] = self.preprocessor_kwargs
            processed_sample["modalities"].append(modality)

        return processed_sample

    def __iter__(self):

        buffer = []
        total_tokens = 0
        N = self.lookahead
        ctx_len = self.ctx_len
        separator_token_idx = self.separator_token_idx
        ignore_index = self.ignore_index

        stream = self.stream

        while True:
            # Collect sequences until we have enough tokens
            while total_tokens < N * ctx_len:
                
                try:
                    sample = next(stream)
                except StopIteration:
                    if not self.infinite:
                        return
                    else:
                        self.stream.reset()

                processed_sample = self.process_sample(sample)
                input_ids = processed_sample['input_ids']
                
                if len(input_ids) > ctx_len:
                    continue
                
                attention_mask = processed_sample['attention_mask']
                labels = processed_sample['labels']
                buffer.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'modalities': processed_sample['modalities'],
                })
                total_tokens += len(input_ids)

            packs : List[List[Dict[str, torch.Tensor]]] = redistribute_batches(buffer, N)
            
            for pack in packs:
                if not pack:
                    continue
                
                input_ids_list = []
                labels_list = []
                attention_mask_list = []
                position_ids_list = []
                modalities_list = []

                # For each sequence in the pack, add the separator and concatenate
                for i, seq in enumerate(pack):
                    input_ids_list.append(seq['input_ids'])
                    labels_list.append(seq['labels'])
                    attention_mask_list.append(seq['attention_mask'])
                    position_ids_list.append(torch.arange(len(seq['input_ids']), dtype=torch.long))
                    modalities_list.append(seq['modalities'])
                
                curr_pack_len = sum([len(seq) for seq in input_ids_list])
                
                while curr_pack_len > ctx_len:
                    input_ids_list.pop()
                    labels_list.pop()
                    attention_mask_list.pop()
                    position_ids_list.pop()
                    modalities_list.pop()
                    curr_pack_len = sum([len(seq) for seq in input_ids_list])                    
                
                if self.shuffle:
                    indices = torch.randperm(len(input_ids_list))
                    
                    input_ids_list = [input_ids_list[i] for i in indices]
                    labels_list = [labels_list[i] for i in indices]
                    attention_mask_list = [attention_mask_list[i] for i in indices]
                    position_ids_list = [position_ids_list[i] for i in indices]
                    modalities_list = [modalities_list[i] for i in indices] 
                    
                input_ids = torch.cat(input_ids_list)
                labels = torch.cat(labels_list)
                attention_mask = torch.cat(attention_mask_list)
                position_ids = torch.cat(position_ids_list)
                modalities = [b for a in modalities_list for b in a]

                # Pad until the maximum context length
                pad_len = ctx_len - len(input_ids)
                input_ids = torch.cat([input_ids, torch.full((pad_len,), separator_token_idx, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), ignore_index, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                position_ids = torch.cat([position_ids, torch.arange(pad_len, dtype=torch.long)])

                # Yield the packed sample
                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask,
                    'modalities': modalities,
                    'position_ids': position_ids
                }

            # Clear the buffer and total_tokens
            buffer = []
            total_tokens = 0
