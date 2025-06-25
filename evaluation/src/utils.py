from typing import List, Tuple, Iterable, Generic, TypeVar
import itertools
import torch
import heapq


def unzip_two(tuples):
    if len(tuples) == 0:
        return [], []
    return list(zip(*tuples))


def expand_attachments(
    attachment_points_ids: List[int],
    embeds_ids,
    attachment_embeds,
    additional_inputs: List[Tuple[any, torch.Tensor]] = []
):
    additional_fill_values, additional_tensors = unzip_two(additional_inputs)

    # Compute the theoretical size of the input_ids tensor after removing the attachment points and adding the modality embeds
    embeds_ids_size = embeds_ids.shape[0] + \
        sum(embed.shape[0] - 1 for embed in attachment_embeds)

    # Create new tensors with the appropriate size for the expanded sequence
    new_embeds_shape = (embeds_ids_size,) + embeds_ids.shape[1:]
    new_embeds_ids = torch.zeros(
        *new_embeds_shape, dtype=embeds_ids.dtype, device=embeds_ids.device)
    new_additional_tensors = [torch.zeros(*((embeds_ids_size,) + tensor.shape[1:]),
                                          dtype=tensor.dtype, device=tensor.device) for tensor in additional_tensors]

    # Variables to track the current insert position and previously processed position
    current_insert_index = 0
    previous_inserted_position = 0

    # Iterate over the attachment points and expand embeddings
    for attachment_index, old_token_index in enumerate(attachment_points_ids):
        # Copy the input_ids and additional inputs up to the attachment point (but not including the attachment token)
        if previous_inserted_position < old_token_index:
            new_embeds_ids[current_insert_index:current_insert_index + old_token_index -
                           previous_inserted_position] = embeds_ids[previous_inserted_position:old_token_index]
            for new_tensor, tensor in zip(new_additional_tensors, additional_tensors):
                new_tensor[current_insert_index:current_insert_index + old_token_index -
                           previous_inserted_position] = tensor[previous_inserted_position:old_token_index]
            current_insert_index += old_token_index - previous_inserted_position

        # Insert the modality embeddings at the attachment point (and remove the attachment token)
        modality_embed = attachment_embeds[attachment_index]
        modality_embed_len = modality_embed.shape[0]
        new_embeds_ids[current_insert_index:current_insert_index +
                       modality_embed_len] = modality_embed
        for fill_value, new_tensor in zip(additional_fill_values, new_additional_tensors):
            new_tensor[current_insert_index:current_insert_index +
                       modality_embed_len].fill_(fill_value)

        # Move past the attachment token
        current_insert_index += modality_embed_len
        previous_inserted_position = old_token_index + 1  # Skip the attachment token

    # Copy the remaining input_ids and additional inputs after the last attachment point
    new_embeds_ids[current_insert_index:] = embeds_ids[previous_inserted_position:]
    for new_tensor, tensor in zip(new_additional_tensors, additional_tensors):
        new_tensor[current_insert_index:] = tensor[previous_inserted_position:]

    # Return the new input_ids and additional inputs
    return new_embeds_ids, new_additional_tensors


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right"
) -> torch.Tensor:
    # Get the maximum length of the sequence
    max_length = max([len(tensor) for tensor in sequences])

    # Transform the sequence by padding on the correct side
    padded_sequences = [
        torch.nn.functional.pad(
            tensor,
            (0, max_length - len(tensor)
             ) if padding_side == "right" else (max_length - len(tensor), 0),
            value=padding_value
        )
        for tensor in sequences
    ]

    # Stack the padded sequences
    return torch.stack(padded_sequences)


def pad_tensor_sequence(
    sequences: List[torch.Tensor],
    padding_value: torch.Tensor,
    padding_side: str = 'right'
):
    # Get the maximum length of the sequence
    max_length = max([len(tensor) for tensor in sequences])

    padded_sequences = []
    for tensor in sequences:
        if len(tensor) - max_length > 0:
            if padding_side == 'left':
                padded_tensor = torch.stack(
                    (padding_value.repeat(len(tensor) - max_length), tensor),
                    dim=0
                )
            else:
                padded_tensor = torch.stack(
                    (tensor, padding_value.repeat(len(tensor) - max_length)),
                    dim=0
                )
        else:
            padded_tensor = tensor

        padded_sequences.append(
            padded_tensor
        )
    return torch.stack(padded_sequences)


T = TypeVar('T')


def batch(input: Iterable[T], batchsize: int) -> Iterable[List[T]]:
    it = iter(input)
    while True:
        batch = list(itertools.islice(it, batchsize))
        if not batch:
            return
        yield batch


def redistribute_batches(all_elements: List, N: int) -> List[List]:
    """
    Args:
        all_elements: List of elements to redistribute
        N: Number of batches to redistribute the elements into
    """

    sorted_strings = sorted(all_elements, key=lambda x: len(x["input_ids"]), reverse=True)

    new_batches = [[] for _ in range(N)]
    heap = [(0, i) for i in range(N)]  # Each item is (current_batch_length, batch_index)
    heapq.heapify(heap)

    # Assign each string to the batch with the minimum current total length
    for s in sorted_strings:
        current_length, idx = heapq.heappop(heap)
        new_batches[idx].append(s)
        heapq.heappush(heap, (current_length + len(s), idx))

    return new_batches 

def string_to_tensor(string: str, tokenizer, max_length: int) -> torch.Tensor:
    tokenized = tokenizer(string, max_length=max_length, truncation=True)
    return torch.tensor(tokenized['input_ids'], dtype=torch.long)

def tensor_to_string(tensor: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(tensor)

def two_way(obj, mode, tokenizer, max_length=None) -> torch.Tensor | str:
    if mode == 'encode':
        return string_to_tensor(obj, tokenizer, max_length)
    elif mode == 'decode':
        return tensor_to_string(obj, tokenizer)
    else:
        raise ValueError(f"Invalid mode: {mode}")
