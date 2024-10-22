# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""pretrain.py"""

from typing import List

import yaml
import numpy as np
import mindspore
from mindspore import Tensor, ops
from mindnlp.transformers import BertForMaskedLM, BertConfig
from tqdm import tqdm

from utils import (
    get_linear_schedule_with_warmup,
    load_text_lines,
    map_lines_to_numbers,
    process_pretrain_dataset,
    setup_optimizer,
)


# ================================
# Forward Propagation Function
# ================================


def forward_fn(
        model: BertForMaskedLM,
        input_ids: Tensor,
        attention_mask: Tensor,
        masked_lm_labels: Tensor,
        vocab_size: int,
) -> Tensor:
    """
    Forward propagation function to compute loss.

    Args:
        model (BertForMaskedLM): BERT model.
        input_ids (Tensor): Input IDs.
        attention_mask (Tensor): Attention mask.
        masked_lm_labels (Tensor): Masked language model labels.
        vocab_size (int): Vocabulary size.

    Returns:
        Tensor: Computed loss.
    """
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    loss = ops.cross_entropy(
        logits.view(-1, vocab_size), masked_lm_labels.view(-1), ignore_index=-100
    )
    return loss


# ================================
# Training Helper Functions
# ================================


def compute_num_train_epochs(
        train_dataloader_length: int,
        total_training_steps: int,
        gradient_accumulation_steps: int,
) -> int:
    """
    Computes the number of training epochs.

    Args:
        train_dataloader_length (int): Length of the training data loader.

    Returns:
        int: Computed number of training epochs.
    """
    steps_per_epoch = train_dataloader_length // gradient_accumulation_steps
    num_train_epochs = total_training_steps // steps_per_epoch + 1
    return num_train_epochs


def save_model_checkpoint(
        model: BertForMaskedLM, epoch: int, checkpoint_dir: str
) -> None:
    """
    Saves the model checkpoint.

    Args:
        model (BertForMaskedLM): BERT model.
        epoch (int): Current epoch number.
    """
    checkpoint_path = f"{checkpoint_dir}bert_model_epoch_{epoch + 1}.ckpt"
    mindspore.save_checkpoint(model, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")


# ================================
# Main Training Function
# ================================


def main() -> None:
    """
    Main training function that executes the model training process.
    """
    # Read configuration file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pretrain_config = config["pretrain"]
    paths_config = config["paths"]

    # mapping vocabulary
    word2idx = map_lines_to_numbers(paths_config["file_vocab_path"])
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(idx2word)

    # Load text data
    lines = load_text_lines(paths_config["file_txt_path"])

    # Create datasets and data loaders
    train_dataset = process_pretrain_dataset(
        lines=lines, word2idx=word2idx, max_seq_length=pretrain_config["max_seq_length"]
    )
    train_dataloader = train_dataset.batch(batch_size=pretrain_config["batch_size"])

    # Configuring the BERT model
    bert_config = BertConfig(vocab_size=vocab_size, type_vocab_size=2)
    model = BertForMaskedLM(config=bert_config)

    # Setup optimizer
    optimizer = setup_optimizer(
        model,
        learning_rate=float(pretrain_config["learning_rate"]),
        epsilon=float(pretrain_config["epsilon"]),
        betas=tuple(float(beta) for beta in pretrain_config["betas"]),  # 转换为浮点数
        weight_decay=float(pretrain_config["weight_decay"]),
    )

    # Initialize the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=pretrain_config["num_warmup_steps"],
        num_training_steps=pretrain_config["total_training_steps"],
    )

    # Define forward propagation and layer computation
    grad_fn = mindspore.value_and_grad(
        lambda input_ids, attention_mask, labels: forward_fn(
            model, input_ids, attention_mask, labels, vocab_size
        ),
        None,
        optimizer.parameters,
        has_aux=False,
    )

    # Calculate the number of training rounds
    num_train_epochs = compute_num_train_epochs(
        train_dataloader_length=len(train_dataloader),
        total_training_steps=pretrain_config["total_training_steps"],
        gradient_accumulation_steps=pretrain_config["gradient_accumulation_steps"],
    )

    # Initialize training variables
    accumulated_grads = None
    losses: List[float] = []
    global_step = 0

    # training cycle
    for epoch in range(num_train_epochs):
        epoch_loss = 0.0
        step_count = 0
        description = f"Epoch {epoch + 1}/{num_train_epochs}"
        with tqdm(total=len(train_dataloader), desc=description, unit="batch") as pbar:
            for step, batch in enumerate(train_dataloader.create_tuple_iterator()):
                input_ids, attention_mask, masked_lm_labels = batch

                # Calculate losses and layers
                loss, grads = grad_fn(input_ids, attention_mask, masked_lm_labels)
                loss_value = loss.asnumpy()

                if np.isnan(loss_value):
                    pbar.update(1)
                    continue

                losses.append(loss_value)
                epoch_loss += loss_value
                step_count += 1

                # Cumulative layer
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = [
                        accum_grad + grad
                        for accum_grad, grad in zip(accumulated_grads, grads)
                    ]

                # When the layer cumulative steps are reached, perform the optimization step
                if (step + 1) % pretrain_config["gradient_accumulation_steps"] == 0:
                    optimizer(accumulated_grads)
                    accumulated_grads = None
                    scheduler.step()
                    global_step += 1

                    # Average print loss per 100 steps
                    if global_step % 100 == 0:
                        avg_loss = np.mean(losses)
                        print(f"\n[Global Step {global_step}] Loss: {avg_loss:.4f}")
                        losses = []

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": loss_value})

        # Calculate and print the average loss per epoch
        avg_epoch_loss = epoch_loss / step_count if step_count > 0 else 0.0
        print(f"\n[Epoch {epoch + 1}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint
        save_model_checkpoint(model, epoch, paths_config["checkpoint_dir"])


if __name__ == "__main__":
    main()
