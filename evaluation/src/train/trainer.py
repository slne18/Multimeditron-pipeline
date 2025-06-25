import torch
from torch.utils.data import DataLoader
from typing import List
from enum import Enum
from src.model.model import MultimodalRawInput
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator
from typing import Optional


class TrainingMode(Enum):
    ALIGNMENT = 0
    END2END = 1
    LM_ONLY = 2

TRAINING_MAPPING = {i.name: i for i in TrainingMode}

class MultimodalTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        accelerator: Optional[Accelerator] = None,
        callbacks=None,
        optimizers=(None, None),
        training_mode: TrainingMode = TrainingMode.ALIGNMENT,
        **kwargs
    ):
        """
        Initializes the trainer.

        Args:
            model: The model to train, evaluate or use for predictions.
            args: The arguments to tweak for training.
            data_collator: The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`.
            train_dataset: The dataset to use for training.
            eval_dataset: The dataset to use for evaluation.
            tokenizer: The tokenizer used to preprocess the data.
            model_init: A function that instantiates the model to be used.
            compute_metrics: A function that will be called at the end of each evaluation phase.
            callbacks: A list of callbacks to customize the training loop.
            optimizers: A tuple containing the optimizer and the scheduler to use.
            training_mode (TrainingMode): The training mode, default to ALIGNMENT.
            **kwargs: Additional keyword arguments to pass to the Trainer.
        """
        # # Initialize the accelerator
        # if accelerator is not None:
        #     optimizer_ = optimizers[0]
        #     model.to(accelerator.device)
        #     model, train_dataset, optimizer_ = accelerator.prepare(
        #         model, train_dataset, optimizer_
        #     )
        #     optimizers = (optimizer_, optimizers[1])

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs
        )
        self.training_mode = training_mode

    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for multimodal inputs.

        Args:
            model: The model to compute the loss.
            inputs: The inputs from the DataLoader.
            return_outputs: Whether or not to return the outputs.

        Returns:
            The loss or (loss, outputs) if return_outputs is True.
        """
        # Prepare multimodal inputs
        multimodal_inputs = []
        multimodal_inputs = inputs["multimodal_inputs"].data

        # Prepare model inputs
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs.get('attention_mask', None),
            'labels': inputs['labels'],
            'multimodal_inputs': multimodal_inputs,
        }

        outputs = model(**model_inputs)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


    def train(self, *args, **kwargs):
        """
        Custom training loop that sets the model in the correct training mode before training.

        Args:
            *args: Positional arguments passed to the Trainer's train method.
            **kwargs: Keyword arguments passed to the Trainer's train method.

        Returns:
            A `TrainOutput` object with training information.
        """
        self.model.train()

        # Set the model in the correct training mode
        if self.training_mode == TrainingMode.ALIGNMENT:
            self.model.freeze_for_alignment()
        elif self.training_mode == TrainingMode.LM_ONLY:
            self.model.freeze_for_lm()
        elif self.training_mode == TrainingMode.END2END:
            self.model.unfreeze()
        else:
            raise ValueError(f"Unknown training mode {self.training_mode}")

        return super().train(*args, **kwargs)
