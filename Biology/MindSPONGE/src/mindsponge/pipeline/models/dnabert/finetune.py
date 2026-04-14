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
"""finetune.py"""

import os

import yaml
import numpy as np
import mindspore
from mindspore import nn, load_checkpoint, load_param_into_net, ops, Tensor
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import BertForSequenceClassification
from tqdm import tqdm

from utils import (
    map_lines_to_numbers,
    DnaDataset,
    process_finetune_dataset,
    setup_finetune_optimizer,
)

def initialize_model(paths_config: dict, finetune_config: dict, word2idx: dict) -> BertForSequenceClassification:
    """
    Initialize the BERT model for sequence classification.

    Parameters:
        paths_config (dict): Paths configuration.
        finetune_config (dict): Fine-tuning configuration.
        word2idx (dict): Vocabulary mapping.

    Returns:
        BertForSequenceClassification: Initialized BERT model.
    """
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=finetune_config["num_labels"]
    )

    custom_vocab_size = len(word2idx)
    model.bert.embeddings.word_embeddings = nn.Embedding(
        custom_vocab_size, model.config.hidden_size
    )
    model.bert.embeddings.word_embeddings.weight.name = (
        "bert.embeddings.word_embeddings.weight"
    )

    param_dict = load_checkpoint(paths_config["pretrained_checkpoint"])
    load_param_into_net(model, param_dict)
    return model

def main() -> None:
    """
    The forward propagation function is used to calculate losses and predictions.
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    finetune_config = config["finetune"]
    paths_config = config["paths"]

    # mapping vocabulary
    word2idx = map_lines_to_numbers(paths_config["file_vocab_path"])

    dataset_train_source = DnaDataset(paths_config["train_dataset_path"])
    dataset_val_source = DnaDataset(paths_config["dev_dataset_path"])

    dataset_train = process_finetune_dataset(
        source=dataset_train_source,
        word2idx=word2idx,
        max_seq_len=finetune_config["max_seq_len"],
        batch_size=finetune_config["batch_size"],
        shuffle=True,
    )
    dataset_val = process_finetune_dataset(
        source=dataset_val_source,
        word2idx=word2idx,
        max_seq_len=finetune_config["max_seq_len"],
        batch_size=finetune_config["batch_size"],
        shuffle=False,
    )

    model = initialize_model(paths_config, finetune_config, word2idx)

    optimizer = setup_finetune_optimizer(
        model,
        learning_rate=float(finetune_config["learning_rate"]),
    )

    def forward_fn(
            input_ids: Tensor,
            attention_mask: Tensor,
            labels: Tensor,
    ):
        """
        Forward function to compute loss and predictions.

        Parameters:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask.
            labels (Tensor): True labels.

        Returns:
            tuple: (loss, logits)
        """
        logits = model(input_ids, attention_mask=attention_mask).logits
        loss = ops.cross_entropy(logits.view(-1, model.num_labels), labels.view(-1))
        return loss, logits

    grad_fn = mindspore.value_and_grad(
        forward_fn,
        None,
        optimizer.parameters,
        has_aux=True,
    )

    def train_step(input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tensor:
        """
        Performs one training step, computes loss, and updates model parameters.

        Parameters:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask.
            labels (Tensor): True labels.

        Returns:
            Tensor: Current loss value.
        """
        (loss, _), grads = grad_fn(input_ids, attention_mask, labels)
        optimizer(grads)
        return loss

    def train_loop(
            model: BertForSequenceClassification, dataset: GeneratorDataset
    ) -> None:
        """
        Training loop that iterates over the dataset to train the model.
        """
        size = dataset.get_dataset_size()
        model.set_train()
        progress_bar = tqdm(
            dataset.create_tuple_iterator(), total=size, desc="Training", unit="batch"
        )
        for input_ids, attention_mask, labels in progress_bar:
            loss = train_step(input_ids, attention_mask, labels)
            loss_np = loss.asnumpy()
            progress_bar.set_postfix({"loss": f"{loss_np:.6f}"})

    def test_loop(
            model: BertForSequenceClassification, dataset: GeneratorDataset
    ) -> None:
        """
        Validation loop that iterates over the validation set to evaluate the model.
        """
        num_batches = dataset.get_dataset_size()
        model.set_train(False)
        total, test_loss, correct = 0, 0, 0
        progress_bar = tqdm(
            dataset.create_tuple_iterator(),
            total=num_batches,
            desc="Validation",
            unit="batch",
        )
        for input_ids, attention_mask, labels in progress_bar:

            input_ids = Tensor(input_ids)
            attention_mask = Tensor(attention_mask)
            labels = Tensor(labels)

            logits = model(input_ids, attention_mask).logits
            loss = ops.cross_entropy(logits.view(-1, 2), labels.view(-1))
            test_loss += loss.asnumpy()

            predictions = np.argmax(logits.asnumpy(), axis=1)
            correct += (predictions == labels.asnumpy()).sum()
            total += labels.shape[0]

            current_loss = loss.asnumpy()
            progress_bar.set_postfix({"current_loss": f"{current_loss:.6f}"})

        test_loss /= num_batches
        correct /= total
        tqdm.write(
            f"Validation Results: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}"
        )

        model.set_train(True)

    for epoch in range(finetune_config["epochs"]):
        tqdm.write(f"Epoch {epoch + 1}/{finetune_config['epochs']}\n{'-' * 30}")
        train_loop(model, dataset_train)
        test_loop(model, dataset_val)
    tqdm.write("Training Complete!")

    os.makedirs(paths_config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(paths_config["checkpoint_dir"], "final_model.ckpt")
    mindspore.save_checkpoint(model, checkpoint_path)
    tqdm.write(f"Model has been saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
