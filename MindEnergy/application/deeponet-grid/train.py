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
Main training script for DeepONet-Grid-UQ
"""
import argparse
import json
import logging
import os
import sys

import mindspore as ms
from mindspore import context
from mindspore.communication import init

from src.data import load_and_preprocess_real_data
from src.model import Prob_DeepONet
from src.trainer import create_trainer
from src.utils import load_config


def train(args, logger):
    """Main training function"""

    config = load_config(args.config)

    config["training"]["distributed"] = args.distributed

    if config["training"]["distributed"] == 0:
        device_id = context.get_context(attr_key="device_id")
    else:
        device_id = int(os.environ.get("MS_NODE_ID"))
    is_main_process = (True if device_id ==
                       0 or config["training"]["distributed"] == 0 else False)

    os.makedirs(config["output"]["save_dir"], exist_ok=True)

    datasets, metadata = load_and_preprocess_real_data(config)

    # create model
    model_config = config["model"]

    m = metadata.get("n_sensors", model_config.get("m", 33))
    dim = model_config.get("dim", 1)
    width = model_config.get("width", 200)
    depth = model_config.get("depth", 3)
    n_basis = model_config.get("n_basis", 100)

    branch_type = model_config.get("branch_type", "modified")
    trunk_type = model_config.get("trunk_type", "modified")
    activation = model_config.get("activation", "sin")

    # Compute layer sizes automatically (following original author's formula)
    # [input] + [width] * depth + [output]
    # depth = 3 means 3 hidden layers
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
        branch=branch_config,
        trunk=trunk_config,
        use_bias=model_config["use_bias"])

    trainer = create_trainer(
        model=model,
        config=config,
        save_dir=config["output"]["save_dir"],
        distributed=config["training"]["distributed"],
    )

    if args.resume:
        trainer.load_model(args.resume)

    if args.eval:
        metrics = trainer.evaluate(datasets["test"])

        results_path = os.path.join(
            config["output"]["save_dir"],
            "test_results.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return

    history = trainer.train(
        train_dataset=datasets["train"], val_dataset=datasets["val"]
    )

    history_path = os.path.join(
        config["output"]["save_dir"],
        "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    test_metrics = trainer.evaluate(datasets["test"])

    test_results_path = os.path.join(
        config["output"]["save_dir"],
        "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    if is_main_process:
        trainer.save_model("final_model.ckpt")
    logger.info(
        "Training completed successfully, model saved to final_model.ckpt")

    logger.info("Final Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeepONet-Grid-UQ model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint")
    parser.add_argument(
        "--eval", action="store_true", help="Only run evaluation on test set"
    )
    parser.add_argument(
        "--distributed", type=int, default=0, help="Distributed training"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    # Data Parallel Init for distributed training
    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="Ascend",
    )
    if args.distributed:
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True
        )
        init()
    ms.set_seed(1)

    train(args, logger)
