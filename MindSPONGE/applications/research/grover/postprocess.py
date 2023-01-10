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
Process the 310 infer results.
"""
import os
import argparse
import numpy as np
import mindspore as ms
from src.data.scaler import StandardScaler
from src.util.utils import get_metric_func


def eval_result():
    """
    Process the results of 310 inferring.
    """
    parser = argparse.ArgumentParser(description='postprocess')
    parser.add_argument('--dataset', type=str, default='bbbp', help='Dataset directory')
    parser.add_argument('--dataset_type', type=str, default='classification', help='Dataset type')
    parser.add_argument('--metrics', type=str, default='auc', help='Metrics')
    parser.add_argument('--result_path', type=str, default='./result_Files', help='Result path')
    parser.add_argument('--label_path', type=str, default='./preprocess_Result', help='Label path')
    parser.add_argument('--scaler_path', type=str, default='../ckpt', help='scaler dir')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    config = parser.parse_args()

    config.label_path = os.path.join(config.label_path, config.dataset, "09_targets")

    # load data
    print('Logging Info - load result data:')
    preds = np.fromfile(os.path.join(config.result_path, config.dataset + '_0.bin'), np.float32).reshape(
        config.batch_size, -1)
    targets = np.fromfile(os.path.join(config.label_path, 'targets.bin'), np.float32).reshape(config.batch_size, -1)
    config.scaler_path = os.path.join(config.scaler_path, config.dataset + "_scaler")
    labels_scaler_path = os.path.join(config.scaler_path, "labels_scaler.ckpt")

    print('Logging Info - Evaluate over test data:')
    preds = preds.tolist()
    targets = targets.tolist()
    num_classes = len(preds[0])
    preds_new = [[] for _ in range(num_classes)]
    targets_new = [[] for _ in range(num_classes)]
    for i in range(num_classes):
        for j, _ in enumerate(preds):
            preds_new[i].append(preds[j][i])
            targets_new[i].append(targets[j][i])

    labels_scaler = None
    if config.dataset_type == "regression":
        state = ms.load_checkpoint(labels_scaler_path)
        labels_scaler = StandardScaler(state['means'].asnumpy(),
                                       state['stds'].asnumpy(),
                                       replace_nan_token=None)

    if labels_scaler is not None:
        preds_new = labels_scaler.inverse_transform(preds_new, num_classes)
        targets_new = labels_scaler.inverse_transform(targets_new, num_classes)

    metric_list = []
    metric_func = get_metric_func(config.metrics)
    for i in range(num_classes):
        metric_list.append(metric_func(targets_new[i], preds_new[i]))
    metric = np.mean(metric_list)

    print(f'Logging Info - test_{config.metrics}: {metric}')


if __name__ == "__main__":
    eval_result()
