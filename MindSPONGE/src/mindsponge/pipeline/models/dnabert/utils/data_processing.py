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
"""utils/data_processing.py"""
import copy
import os
from typing import Any, Dict, List, Tuple

import yaml
import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype
from mindspore.dataset import GeneratorDataset, transforms

from .constants import (
    UNKNOWN_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    COLUMN_NAMES_PRETRAIN,
)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
pretrain_config = config["pretrain"]


def map_lines_to_numbers(file_path: str) -> Dict[str, int]:
    """
    Maps each line in the file to a unique numeric index.

    Args:
        file_path (str): Path to the vocabulary file.

    Returns:
        Dict[str, int]: Mapping from words to indices.

    Raises:
        FileNotFoundError: Vocabulary file not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vocabulary file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    word2idx = {line.strip(): i for i, line in enumerate(lines)}
    return word2idx


def load_text_lines(file_path: str) -> List[str]:
    """
    Loads non-empty lines from a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        List[str]: List of non-empty lines.

    Raises:
        FileNotFoundError: Text file not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines


class PreDataset:
    """
    Custom dataset for generating input data for BERT.

    Args:
        lines (List[str]): List of text lines.
        word2idx (Dict[str, int]): Mapping from words to indices.
        max_seq_length (int, optional): Maximum sequence length. Default is `MAX_SEQ_LENGTH`.

    Attributes:
        word2idx (Dict[str, int]): Mapping from words to indices.
        max_seq_length (int): Maximum sequence length.
        sentences (List[List[int]]): List of processed sentence indices.
    """

    def __init__(
            self, lines, word2idx, max_seq_length=pretrain_config["max_seq_length"]
    ):
        """
        Initializes the dataset.

        Args:
            lines (List[str]): List of text lines.
            word2idx (Dict[str, int]): Mapping from words to indices.
            max_seq_length (int, optional): Maximum sequence length. Default is `MAX_SEQ_LENGTH`.
        """
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length
        self.sentences = self._process_lines(lines)

    def _process_lines(self, lines):
        """
        Processes text lines by converting words to indices.

        Args:
            lines (List[str]): List of text lines.

        Returns:
            List[List[int]]: List of converted sentence indices.
        """
        sentences = []
        for line in lines:
            words = line.split()
            sentence = [
                self.word2idx.get(word, self.word2idx.get(UNKNOWN_TOKEN, 0))
                for word in words
            ]
            sentences.append(sentence)
        return sentences

    def __len__(self):
        """
        Gets the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Gets the data at the specified index.

        Args:
            idx (int or Tensor): Data index.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (input_ids, attention_mask, masked_lm_labels)

        Raises:
            IndexError: Index out of range.
        """
        if isinstance(idx, mindspore.Tensor):
            idx = idx.asnumpy().item()
        else:
            idx = int(idx)

        sentence = self.sentences[idx]
        word2idx = self.word2idx

        # Add special tags
        tokens = [word2idx.get(CLS_TOKEN, 0)] + sentence + [word2idx.get(SEP_TOKEN, 0)]

        if len(tokens) > self.max_seq_length:
            tokens = tokens[: self.max_seq_length]

        input_ids = tokens + [word2idx.get(PAD_TOKEN, 0)] * (
            self.max_seq_length - len(tokens)
        )
        input_ids = np.array(input_ids, dtype=np.int32)

        # Create an attention mask
        attention_mask = np.where(input_ids != word2idx.get(PAD_TOKEN, 0), 1, 0).astype(
            np.int32
        )

        # Create mask tag
        masked_lm_labels = input_ids.copy()

        # Create a mask probability matrix
        probability_matrix = np.full(
            input_ids.shape, pretrain_config["mask_probability"], dtype=np.float32
        )
        special_tokens = [
            word2idx.get(CLS_TOKEN, 0),
            word2idx.get(SEP_TOKEN, 0),
            word2idx.get(MASK_TOKEN, 0),
            word2idx.get(PAD_TOKEN, 0),
        ]
        probability_matrix[np.isin(input_ids, special_tokens)] = 0.0

        # randomly select the mask position
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)

        # Extended Mask Range
        end = (
            np.max(np.where(probability_matrix != 0)[0])
            if np.any(probability_matrix != 0)
            else 0
        )
        mask_centers = set(np.where(masked_indices)[0])
        new_centers = copy.deepcopy(mask_centers)
        for center in mask_centers:
            for offset in pretrain_config["mask_list"]:
                current_index = center + offset
                if 1 <= current_index <= end:
                    new_centers.add(current_index)
        new_centers = list(new_centers)
        masked_indices[new_centers] = True

        # Set the unmasked position to -100.
        masked_lm_labels[~masked_indices] = -100

        # Replace the partial mask with the [MASK] tag
        indices_replaced = (
            np.random.binomial(1, 0.8, size=input_ids.shape).astype(bool)
            & masked_indices
        )
        input_ids[indices_replaced] = word2idx.get(MASK_TOKEN, 0)

        # Random replacement partial mask to random vocabulary
        indices_random = (
            np.random.binomial(1, 0.5, size=input_ids.shape).astype(bool)
            & masked_indices
            & ~indices_replaced
        )
        random_tokens = np.random.randint(5, len(word2idx), size=input_ids.shape)
        input_ids[indices_random] = random_tokens[indices_random]

        # Convert to MindSpore tensor
        input_ids = Tensor(input_ids, mstype.int32)
        attention_mask = Tensor(attention_mask, mstype.int32)
        masked_lm_labels = Tensor(masked_lm_labels, mstype.int32)

        return input_ids, attention_mask, masked_lm_labels


class DnaDataset:
    """
    The DnaDataset class is used to load and manage DNA sequence datasets.

    Parameters:
        path (str): Path to the dataset file.

    Exceptions:
        FileNotFoundError: Raises this exception if the dataset file is not found.
    """

    def __init__(self, path: str):
        self.path = path
        self._sequence = []
        self._labels = []
        self._load()

    def _load(self):
        """
        Loads the dataset file and parses sequences and labels.

        Exceptions:
            FileNotFoundError: Raises this exception if the dataset file is not found.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        with open(self.path, "r", encoding="utf-8") as file:
            dataset = file.read()
        lines = dataset.strip().split("\n")
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            sequence, label = parts
            self._sequence.append(sequence)
            self._labels.append(int(label))

    def __getitem__(self, index: Any) -> Tuple[str, int]:
        """
        Retrieves data at the specified index.

        Parameters:
            index (int): Index of the data.

        Returns:
            tuple: (sequence, label)
        """
        return self._sequence[index], self._labels[index]

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self._labels)


def process_pretrain_dataset(
        lines: List[str], word2idx: Dict[str, int], max_seq_length: int
) -> GeneratorDataset:
    """
    Processing the pre-trained dataset.

    Args:
        lines (List[str]): List of text lines.
        word2idx (Dict[str, int]): Mapping from vocabulary to index.
        max_seq_length (int): Maximum sequence length.

    Returns:
        GeneratorDataset: The pre-trained dataset.
    """
    dataset = PreDataset(lines, word2idx, max_seq_length=max_seq_length)
    generator_dataset = GeneratorDataset(
        dataset,
        column_names=COLUMN_NAMES_PRETRAIN,
        shuffle=True,
    )
    return generator_dataset


def process_finetune_dataset(
        source: DnaDataset,
        word2idx: Dict[str, int],
        max_seq_len: int,
        batch_size: int,
        shuffle: bool,
) -> GeneratorDataset:
    """
    Function to process the dataset.

    Parameters:
        source (DnaDataset): Data source.
        word2idx (dict): Mapping from words to indices.
        max_seq_len (int, optional): Maximum sequence length. Default is 128.
        batch_size (int, optional): Batch size. Default is 8.
        shuffle (bool, optional): Whether to shuffle data. Default is True.

    Returns:
        mindspore.dataset: The processed dataset.
    """
    dataset = source
    generator_dataset = GeneratorDataset(
        dataset, column_names=["sequence", "label"], shuffle=shuffle
    )

    type_cast_op = transforms.TypeCast(mindspore.int32)

    def tokenize_and_pad(text):
        """
        Word segmentation and padding functions.

        parameter：
            - text (str or numpy.ndarray) – Enter text。

        return：
            Tuple: Input ID and attention mask after word segmentation.
        """
        # Examine and convert data types
        if isinstance(text, np.ndarray):
            text = text.item()  # Extract a single element from a numpy array
            if isinstance(text, bytes):
                text = text.decode("utf-8")  # If it is a byte type, decode it
            else:
                text = str(text)  # Otherwise, convert to string.

        words = text.split()
        sentence = [
            word2idx.get(word, word2idx.get(UNKNOWN_TOKEN, 0)) for word in words
        ]

        tokens = [word2idx.get(CLS_TOKEN, 0)] + sentence + [word2idx.get(SEP_TOKEN, 0)]

        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        input_ids = tokens + [word2idx.get(PAD_TOKEN, 0)] * (max_seq_len - len(tokens))
        input_ids = np.array(input_ids, dtype=np.int32)
        attention_mask = np.where(input_ids != word2idx.get(PAD_TOKEN, 0), 1, 0).astype(
            np.int32
        )

        return input_ids, attention_mask

    generator_dataset = generator_dataset.map(
        operations=tokenize_and_pad,
        input_columns="sequence",
        output_columns=["input_ids", "attention_mask"],
    )

    generator_dataset = generator_dataset.map(
        operations=type_cast_op, input_columns="label", output_columns="labels"
    )

    generator_dataset = generator_dataset.batch(batch_size)

    return generator_dataset
