# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
The general utility functions.
"""
import csv
import os
import stat
import math
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, mean_absolute_error, r2_score, \
    recall_score


def accuracy(targets: List[float], preds: List[int], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return accuracy_score(targets, hard_preds)


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def recall(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return recall_score(targets, hard_preds)


def get_metric_func(metric: str):
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    # Note: If you want to add a new metric, please also update the parser argument --metric in parsing.py.
    if metric == 'auc':
        return roc_auc_score

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'recall':
        return recall

    raise ValueError(f'Metric "{metric}" not supported.')


class GroverMetric:
    """
    Metric method
    """

    def __init__(self, args, labels_scaler, save_path=None):
        super(GroverMetric, self).__init__()
        self.num_classes = args.num_tasks
        self.metrics = args.metrics
        self.preds = [[] for _ in range(self.num_classes)]
        self.targets = [[] for _ in range(self.num_classes)]
        self.smiles = []
        self.labels_scaler = labels_scaler
        self.save_path = save_path

    def clear(self):
        """Clear the internal evaluation result."""
        self.preds = []
        self.targets = []
        self.smiles = []

    def update(self, preds, targets, smiles):
        """Update the values."""
        preds = preds.asnumpy().tolist()
        targets = targets.asnumpy().tolist()
        for j in range(len(preds)):
            line = smiles[j].decode('utf-8')
            self.smiles.append(line)

        for i in range(self.num_classes):
            for j, _ in enumerate(preds):
                # if targets[j][i] is not None:  # Skip those without targets
                self.preds[i].append(preds[j][i])
                self.targets[i].append(targets[j][i])

    def denormalize(self):
        if self.labels_scaler is not None:
            self.preds = self.labels_scaler.inverse_transform(self.preds, self.num_classes)
            self.targets = self.labels_scaler.inverse_transform(self.targets, self.num_classes)

    def save(self):
        """Save preds."""
        lines = []
        for i in range(len(self.smiles)):
            preds = [self.preds[j][i] for j in range(self.num_classes)]
            targets = [self.targets[j][i] for j in range(self.num_classes)]
            lines.append([self.smiles[i], preds, targets])
        preds_path = os.path.join(self.save_path, "preds.csv")

        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(preds_path, flags, modes), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(lines)

    def eval(self):
        self.denormalize()
        self.save()
        metric_list = []
        metric_func = get_metric_func(self.metrics)
        for i in range(self.num_classes):
            metric_list.append(metric_func(self.targets[i], self.preds[i]))
        metric = np.mean(metric_list)
        return metric


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the values.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1


def load_smiles_labels(path):
    """
    Load smiles and labels.
    """
    smiles_list = []
    labels_list = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for _, line in enumerate(reader):
            smiles = line[0]
            class_labels = [float(x) if x != '' else 0.0 for x in line[1:]]
            class_labels = np.array(class_labels).astype(np.float32)
            smiles_list.append(smiles)
            labels_list.append(class_labels)
    return smiles_list, labels_list


def load_smiles(path):
    """
    Load lines(containing smiles and labels).
    param path: Path to the file containing smiles
    :return: A list of smiles
    """
    smiles = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for _, line in enumerate(reader):
            smiles.append(line)

    return smiles


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.

    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)
