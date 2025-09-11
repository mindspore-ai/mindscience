#!/usr/bin/env python3
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
# ==============================================================================

"""
Main inference script for DeepONet-Grid-UQ
"""
import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np
from mindspore import context

from src.data import (
    load_and_preprocess_real_data,
    load_real_data,
    trajectory_prediction,
)
from src.metrics import MetricsCalculator
from src.model import Prob_DeepONet
from src.utils import load_config, load_trained_model, plot_pred_UQ


def inference_on_dataset(
    model: Prob_DeepONet,
    dataset_path: str,
    output_dir: str = "inference_results",
) -> None:
    """Perform inference on entire dataset"""
    datasets, metadata = load_and_preprocess_real_data(
        {
            "data": {
                "data_path": dataset_path,
                "use_synthetic": False,
                "train_ratio": 0.0,
                "val_ratio": 0.0,
                "test_ratio": 1.0,
            },
            "training": {
                "batch_size": 1024,
                "distributed": 0,
            },
        }
    )

    os.makedirs(output_dir, exist_ok=True)

    test_dataset = datasets["test"]

    all_predictions = []
    all_targets = []
    all_means = []
    all_stds = []

    for u, y, target in test_dataset:
        mean_pred, log_std_pred = model(u, y)
        std_pred = ms.ops.Exp()(log_std_pred)

        all_predictions.append(mean_pred)
        all_targets.append(target)
        all_means.append(mean_pred)
        all_stds.append(std_pred)

    if all_predictions:
        predictions = ms.ops.Concat(axis=0)(all_predictions)
        targets = ms.ops.Concat(axis=0)(all_targets)
        means = ms.ops.Concat(axis=0)(all_means)
        stds = ms.ops.Concat(axis=0)(all_stds)

        predictions_np = predictions.asnumpy()
        targets_np = targets.asnumpy()
        means_np = means.asnumpy()
        stds_np = stds.asnumpy()

        results = {
            "predictions": predictions_np.tolist(),
            "targets": targets_np.tolist(),
            "means": means_np.tolist(),
            "stds": stds_np.tolist(),
        }

        results_path = os.path.join(output_dir, "inference_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Inference results saved to: {results_path}")


def inference(args, logger):
    """Main inference function"""

    config = load_config(args.config)

    model_config = config["model"]

    m = model_config.get("m", 33)
    dim = model_config.get("dim", 1)
    width = model_config.get("width", 200)
    depth = model_config.get("depth", 3)
    n_basis = model_config.get("n_basis", 100)

    branch_type = model_config.get("branch_type", "modified")
    trunk_type = model_config.get("trunk_type", "modified")
    activation = model_config.get("activation", "sin")

    branch_layer_size = [m] + [width] * depth + [n_basis]
    trunk_layer_size = [dim] + [width] * depth + [n_basis]

    branch_config = {
        "type": branch_type,
        "layer_size": branch_layer_size,
        "activation": activation,
    }

    trunk_config = {
        "type": trunk_type,
        "layer_size": trunk_layer_size,
        "activation": activation,
    }

    model = Prob_DeepONet(
        branch=branch_config, trunk=trunk_config, use_bias=model_config["use_bias"]
    )
    model = load_trained_model(model, args.checkpoint)
    model.set_train(False)

    if args.trajectory_prediction:
        # Test single sample trajectory prediction

        if args.data_path is None:
            logger.error("For single inference, --data_path must be provided")
            return

        u, y, s = load_real_data(args.data_path)

        u_single = u[args.data_index]
        y_single = y[args.data_index]
        s_true = s[args.data_index]

        predictions, std_predictions = trajectory_prediction(u_single, y_single, model)
        s_mean = predictions.reshape(-1)
        s_std = std_predictions.reshape(-1)

        calculator = MetricsCalculator()
        targets = ms.Tensor(s[args.data_index], ms.float32)
        predicts = ms.Tensor(predictions, ms.float32)
        print(f"    Targets: {s[args.data_index].flatten()}")
        print(f"    Predictions: {predictions.flatten()}")

        # Test L1 and L2 relative errors

        l1_error, l2_error = calculator.trajectory_rel_error(targets, predicts)
        metrics_text = f"L1: {l1_error:.4f}\nL2: {l2_error:.4f}"
        print(metrics_text)

        out_dir = (
            args.output_dir if hasattr(args, "output_dir") else "inference_results"
        )
        save_path = os.path.join(
            out_dir, f"trajectory_prediction_idx_{args.data_index}.png"
        )
        test_y = [i + len(u_single) for i in range(len(targets))]

        plot_pred_UQ(
            sensors=np.arange(u_single.shape[0]),
            u=u_single,
            y=test_y,
            s=s_true,
            s_mean=s_mean,
            s_std=s_std,
            save_path=save_path,
        )

    else:
        # Dataset inference
        if args.data_path is None:
            logger.error("For dataset inference, --data_path must be provided")
            return

        # Perform inference on dataset
        inference_on_dataset(model, args.data_path, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DeepONet-Grid-UQ Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/best_model.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/test-data-voltage-m-33-mix.npz",
        help="Path to data file for dataset inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--trajectory_prediction",
        action="store_true",
        help="Perform single data point (with multiple time points) inference",
    )
    parser.add_argument(
        "--data_index",
        type=int,
        default=0,
        help="Index of data point for single inference (default: 0)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("inference.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    inference(args, logger)
