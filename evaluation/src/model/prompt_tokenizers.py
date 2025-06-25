import abc
from transformers import PreTrainedTokenizerBase
from typing import Dict, Any, List, Tuple
import torch
import copy


class PromptTokenizer(abc.ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 modalities_num_embeddings: Dict[str, int],
                 attachment_token_idx: int,
                 ignore_index: int = -100):
        """
        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
            modalities_num_embeddings (Dict[str, int]): A dictionary mapping modality names to the number of embeddings they have.
            ignore_index (int, optional): The index to ignore. Defaults to -100.
            attachment_token (str, optional): The token to use for attachment. Defaults to "<|attachment|>".
        """
        
        self.modalities_num_embeddings = modalities_num_embeddings
        self.tokenizer = copy.deepcopy(tokenizer)
        self.ignore_index = ignore_index
        self.attachment_token_idx = attachment_token_idx

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def tokenize_conversation(self, prompt: List[Dict[str, str]], modalities: List[Dict[str, Any]], add_eos_token=True, add_generation_prompt=False) -> Dict[str, Any]:
        res = self._tokenize_conversation(prompt, modalities, add_eos_token=add_eos_token, add_generation_prompt=add_generation_prompt)
        self.validate_tokenized_results(res)
        return res
    
    def tokenize_text(self, prompt: str, modalities: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = self._tokenize_text(prompt, modalities)
        self.validate_tokenized_results(res)
        return res
        
    def validate_tokenized_results(self, res: Dict[str, Any]):
        if not ("input_ids" in res and "attention_mask" in res and "labels" in res):
            raise ValueError(
                "Result of tokenize_conversation must contain keys: input_ids, attention_mask and labels")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def expand_attachment_input_tokens(self, token_ids: torch.Tensor, attention_mask: torch.Tensor,
                                       modalities_for_message: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expands attachment tokens in the token sequence based on the number of embeddings for each modality.

        Args:
            token_ids (torch.Tensor): The original sequence of token IDs.
            attention_mask (torch.Tensor): The attention mask corresponding to the token_ids.
            modalities_for_message (List[Dict[str, Any]]): A list of modality dictionaries, each containing modality information.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The expanded token IDs and corresponding attention mask.
        """
        if len(modalities_for_message) == 0:
            return token_ids, attention_mask

        modalities_indices = torch.argwhere(
            token_ids == self.attachment_token_idx).flatten()
        modalities_names = list(
            map(lambda x: x["type"], modalities_for_message))

        assert len(modalities_names) == len(modalities_indices)
        assert len(attention_mask) == len(token_ids)

        # First, take all the text until the first modality (excluded)
        expanded_token_ids = [token_ids[: modalities_indices[0] - 1]]
        expanded_attention_mask = [attention_mask[: modalities_indices[0] - 1]]

        # Add the first modality
        num_embeddings = self.modalities_num_embeddings[modalities_names[0]]
        expanded_token_ids.append(torch.tensor(
            [self.attachment_token_idx] * num_embeddings))
        expanded_attention_mask.append(torch.tensor([True] * num_embeddings))

        for previous_mod_idx, current_mod_idx, mod_name in zip(modalities_indices, modalities_indices[1:], modalities_names[1:]):
            # Add the text
            expanded_token_ids.append(
                token_ids[previous_mod_idx + 1: current_mod_idx])
            expanded_attention_mask.append(
                attention_mask[previous_mod_idx + 1: current_mod_idx])

            # Add the correct number of attachment tokens for the current modality
            num_embeddings = self.modalities_num_embeddings[mod_name]
            expanded_token_ids.append(torch.tensor(
                [self.attachment_token_idx] * num_embeddings))
            # Don't want to mask the attachment
            expanded_attention_mask.append(
                torch.tensor([True] * num_embeddings))

        # Add final piece of text
        last_mod_idx = modalities_indices[-1]
        expanded_token_ids.append(token_ids[last_mod_idx + 1:])
        expanded_attention_mask.append(attention_mask[last_mod_idx + 1:])

        expanded_token_ids = torch.cat(expanded_token_ids)
        expanded_attention_mask = torch.cat(expanded_attention_mask)

        return expanded_token_ids, expanded_attention_mask


class Llama3PromptTokenizer(PromptTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 modalities_num_embeddings: Dict[str, int],
                 attachment_token_idx: int,
                 ignore_index: int = -100):
        super().__init__(
            tokenizer=tokenizer,
            modalities_num_embeddings=modalities_num_embeddings,
            ignore_index=ignore_index,
            attachment_token_idx=attachment_token_idx
        )
    
    def _tokenize_text(self, text: str, modalities: List[Dict[str, Any]]) -> Dict[str, Any]:
        outputs = self.tokenizer(text, return_tensors="pt")
        input_ids, attention_mask = self.expand_attachment_input_tokens(
            token_ids=outputs["input_ids"].flatten(),
            attention_mask=outputs["attention_mask"].flatten(),
            modalities_for_message=modalities
        )
        
        # Handle labels
        labels = torch.where(input_ids == self.attachment_token_idx, self.ignore_index, input_ids)
        
        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "labels" : labels
        }
    

    def _tokenize_conversation(self, conversation: List[Dict[str, str]], modalities: List[Dict[str, Any]], 
                               add_eos_token=True, add_generation_prompt=False) -> Dict[str, Any]:
        outputs = self.tokenizer.apply_chat_template(
                conversation, add_eos_token=add_eos_token,
                return_dict=True, return_tensors="pt", 
                add_generation_prompt=add_generation_prompt)
        
        self.expand_attachment_input_tokens(
            token_ids=outputs["input_ids"].flatten(),
            attention_mask=outputs["attention_mask"].flatten(),
            modalities_for_message=modalities
        )
                
        # Expand the attachments
        input_ids, attention_mask = self.expand_attachment_input_tokens(
            token_ids=outputs["input_ids"].flatten(),
            attention_mask=outputs["attention_mask"].flatten(),
            modalities_for_message=modalities
        )
        
        # Handle labels
        labels = torch.clone(input_ids)

        left_tag = self.tokenizer.encode("<|start_header_id|>system<|end_header_id|>", add_special_tokens=False)
        right_tag = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)        
        replace_between_tags(labels, left_tag=left_tag, right_tag=right_tag)
        
        left_tag = self.tokenizer.encode("<|start_header_id|>human<|end_header_id|>", add_special_tokens=False)
        replace_between_tags(labels, left_tag=left_tag, right_tag=right_tag)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def replace_between_tags(tensor, left_tag, right_tag, replace_value=-100):
    tensor_len = tensor.size(0)
    left_tag_len = len(left_tag)
    right_tag_len = len(right_tag)
    
    # Convert left_tag and right_tag to tensors
    left_tag = torch.tensor(left_tag)
    right_tag = torch.tensor(right_tag)

    i = 0
    while i <= tensor_len - left_tag_len:
        # Check if left_tag matches at the current position
        if torch.equal(tensor[i:i + left_tag_len], left_tag):
            # Find the position of right_tag after the left_tag
            for j in range(i + left_tag_len, tensor_len - right_tag_len + 1):
                if torch.equal(tensor[j:j + right_tag_len], right_tag):
                    # Replace all values between the end of left_tag and the start of right_tag with -100
                    tensor[i: j + right_tag_len] = replace_value
                    i = j + right_tag_len - 1  # Move index to the end of the right_tag
                    break
        i += 1
