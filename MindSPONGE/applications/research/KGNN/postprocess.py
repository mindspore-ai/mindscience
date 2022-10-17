"""
postprocess script
"""

# --result_path=./result_Files --dataset=$dataset_name

import os
import argparse
import numpy as np
import sklearn.metrics as m
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve


def eval_result():
    """eval_result"""
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--dataset', type=str, default='kegg', help='Dataset directory')
    parser.add_argument('--result_path', type=str, default='./result_Files', help='Result path')
    parser.add_argument('--label_path', type=str, default='./preprocess_Result/01_data', help='Label path')
    config = parser.parse_args()

    # load data
    print('Logging Info - load result data:')
    y_pred = np.fromfile(os.path.join(config.result_path, config.dataset + '_0.bin'), np.float32)
    y_true = np.fromfile(os.path.join(config.label_path, config.dataset + '_label.bin'), np.int64)

    print('Logging Info - Evaluate over test data:')
    threshold = 0.5
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    p, r, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    aupr = m.auc(r, p)
    y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print(f'Logging Info - test_auc: {auc}, test_acc: {acc}, test_f1: {f1}, test_aupr: {aupr}')


if __name__ == "__main__":
    eval_result()
