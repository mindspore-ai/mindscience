# Copyright 2025 Huawei Technologies Co., Ltd
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

"""classifier_utils script"""
import random
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import perturber_utils as pu
logger = logging.getLogger(__name__)


def label_classes(data, gene_class_dict, nproc):
    """remove cells without any of the target genes"""
    def if_contains_label(example):
        a = pu.flatten_list(gene_class_dict.values())
        b = example["input_ids"]
        return not set(a).isdisjoint(b)
    data = data.filter(if_contains_label, num_proc=nproc)
    if data is None:
        logger.error(
            "No cells remain after filtering for target genes. Check target gene list."
        )
    label_set = gene_class_dict.keys()
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
    id_class_dict = {v: k for k, v in class_id_dict.items()}

    def classes_to_ids(example):
        example["labels"] = label_gene_classes(
            example, class_id_dict, gene_class_dict
            )
        return example
    data = data.map(classes_to_ids, num_proc=nproc)
    return data, id_class_dict


def downsample_and_shuffle(data, max_ncells, max_ncells_per_class, cell_state_dict):
    """downsample and shuffle datasets"""
    data = data.shuffle(seed=42)
    num_cells = len(data)
    # if max number of cells is defined, then subsample to this max number
    if max_ncells:
        if num_cells > max_ncells:
            data = data.select([i for i in range(max_ncells)])
    if max_ncells_per_class:
        class_labels = data[cell_state_dict["state_key"]]
        random.seed(42)
        subsample_indices = subsample_by_class(class_labels, max_ncells_per_class)
        data = data.select(subsample_indices)
    return data


def remove_cols(data, cols_to_keep):
    """remove and cols datasets"""
    other_cols = list(data.features.keys())
    other_cols = [ele for ele in other_cols if ele not in cols_to_keep]
    data = data.remove_columns(other_cols)
    return data


def validate_and_clean_cols(train_data, eval_data):
    """validate and clean cols datasets"""
    # validate that data has expected label column and remove others
    label_col = "labels"

    cols_to_keep = [label_col] + ["input_ids", "length"]
    if label_col not in train_data.column_names:
        logger.error("train_data must contain column %s with class labels.", label_col)
    else:
        train_data = remove_cols(train_data, cols_to_keep)

    if eval_data:
        if label_col not in eval_data.column_names:
            logger.error(
                "eval_data must contain column %s with class labels.", label_col
            )
        else:
            eval_data = remove_cols(eval_data, cols_to_keep)
    return train_data, eval_data


def label_gene_classes(example, class_id_dict, gene_class_dict):
    """label gene classes"""
    return [
        class_id_dict.get(gene_class_dict.get(token_id, -100), -100)
        for token_id in example["input_ids"]
    ]


def gene_split_data(
        data,
        targets,
        labels,
        train_index,
        eval_index,
        max_ncells,
        iteration_num,
        num_proc,
        balance=False,
):
    """split gene data"""
    # generate cross-validation splits
    train_data = gene_classifier_split(
        data,
        targets,
        labels,
        train_index,
        "train",
        max_ncells,
        iteration_num,
        num_proc,
        balance,
    )
    eval_data = gene_classifier_split(
        data,
        targets,
        labels,
        eval_index,
        "eval",
        max_ncells,
        iteration_num,
        num_proc,
        balance,
    )
    return train_data, eval_data


def gene_classifier_split(
        data,
        targets,
        labels,
        index,
        subset_name,
        max_ncells,
        iteration_num,
        num_proc,
        balance=False,
):
    """split gene classifier"""
    # generate cross-validation splits
    targets = np.array(targets)
    labels = np.array(labels)
    targets_subset = targets[index]
    labels_subset = labels[index]
    label_dict_subset = dict(zip(targets_subset, labels_subset))

    # function to filter by whether contains train or eval labels
    def if_contains_subset_label(example):
        a = targets_subset
        b = example["input_ids"]
        return not set(a).isdisjoint(b)

    # filter dataset for examples containing classes for this split
    logger.info("Filtering data for %s genes in split %d", subset_name, iteration_num)
    subset_data = data.filter(if_contains_subset_label, num_proc=num_proc)
    percentage_filtered = round((1 - len(subset_data) / len(data)) * 100)
    logger.info(
        "Filtered %d%%; %d remain\n", percentage_filtered, len(subset_data)
    )

    # balance gene subsets if train
    if (subset_name == "train") and (balance is True):
        subset_data, label_dict_subset = balance_gene_split(
            subset_data, label_dict_subset, num_proc
        )

    # subsample to max_ncells
    subset_data = downsample_and_shuffle(subset_data, max_ncells, None, None)

    # relabel genes for this split
    def subset_classes_to_ids(example):
        example["labels"] = [
            label_dict_subset.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    subset_data = subset_data.map(subset_classes_to_ids, num_proc=num_proc)
    return subset_data


class StratifiedKFold3(StratifiedKFold):
    """StratifiedKFold3"""
    def split(self, targets, labels, test_ratio=0.5, groups=None):
        """split"""
        s = super().split(targets, labels, groups)
        for train_indxs, test_indxs in s:
            if test_ratio == 0:
                yield train_indxs, test_indxs, None
            else:
                labels_test = np.array(labels)[test_indxs]
                valid_indxs, test_indxs = train_test_split(
                    test_indxs,
                    stratify=labels_test,
                    test_size=test_ratio,
                    random_state=0,
                )
                yield train_indxs, valid_indxs, test_indxs


def load_data(dataset):
    """load data"""
    input_ids = dataset['input_ids']
    length = dataset['length']
    labels = dataset['labels']
    return input_ids, length, labels
