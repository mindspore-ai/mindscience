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
Trainer for DeepONet-Grid-UQ
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import Callback

from .metrics import (
    MetricsCalculator,
    compute_mae,
    compute_metrics,
    compute_mse,
    compute_r2_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomCallback(Callback):
    """Custom callback for training monitoring"""

    def __init__(self, log_interval: int = 10, eval_interval: int = 100):
        super(CustomCallback, self).__init__()
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.step = 0
        self.losses = []

    def step_end(self, run_context):
        """Called at the end of each step"""
        self.step += 1
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (list, tuple)):
            loss = loss[0]

        self.losses.append(float(loss))

        if self.step % self.log_interval == 0:
            logger.info("Step %d, Loss: %.6f", self.step, float(loss))


class ProbabilisticLoss(nn.Cell):
    """Negative log likelihood loss for probabilistic DeepONet"""

    def __init__(self):
        super(ProbabilisticLoss, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.square = ops.Square()
        self.reduce_mean = ops.ReduceMean()
        self.log_2pi = np.log(2 * np.pi)

    def construct(self, mean_pred, log_std_pred, target):
        """Compute negative log likelihood loss using normal distribution"""
        std_pred = self.exp(log_std_pred)

        log_prob = (
            -0.5 * self.square((target - mean_pred) / std_pred)
            - 0.5 * log_std_pred
            - 0.5 * self.log_2pi
        )

        return -self.reduce_mean(log_prob)


class MSELoss(nn.Cell):
    """Mean squared error loss"""

    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def construct(self, mean_pred, _, target):
        """Compute MSE loss (ignores uncertainty predictions)"""
        return self.mse(mean_pred, target)


class DeepONetTrainer:
    """Trainer class for DeepONet with uncertainty quantification"""

    def __init__(
        self,
        model: nn.Cell,
        config: Dict[str, Any],
        save_dir: str = "outputs",
        distributed: int = 0,
    ):

        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.training_config = config["training"]

        # Initialize optimizer
        optimizer_type = self.training_config.get("optimizer", "adam")
        if self.training_config.get("scheduler", None) == "cosine":
            min_lr = float(self.training_config.get("min_lr", 1e-7))
            max_lr = float(self.training_config.get("max_lr", 1e-3))
            total_step = int(self.training_config.get("total_step", 10000))
            step_per_epoch = int(
                self.training_config.get("step_per_epoch", 1000))
            decay_epoch = int(self.training_config.get("decay_epoch", 10))
            learning_rate = nn.cosine_decay_lr(
                min_lr, max_lr, total_step, step_per_epoch, decay_epoch
            )  # Ensure it's a float
        else:
            learning_rate = float(
                self.training_config["learning_rate"]
            )  # Ensure it's a float
        weight_decay = float(self.training_config.get("weight_decay", 0.0))

        if optimizer_type.lower() == "adam":
            self.optimizer = nn.Adam(
                self.model.trainable_params(),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        if distributed:
            self.grad_reducer = nn.DistributedGradReducer(
                self.optimizer.parameters)
        else:
            self.grad_reducer = nn.Identity()

        # Initialize loss function
        loss_type = self.training_config.get("loss_type", "nll")

        if loss_type.lower() == "nll":
            self.loss_fn = ProbabilisticLoss()
        elif loss_type.lower() == "mse":
            self.loss_fn = MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Initialize metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "best_loss": float("inf"),
            "patience_counter": 0,
        }

        self.exp = ops.Exp()

    def train_step(
        self, u: ms.Tensor, y: ms.Tensor, target: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Single training step"""

        def forward_fn():
            mean_pred, log_std_pred = self.model(u, y)
            loss = self.loss_fn(mean_pred, log_std_pred, target)
            return loss, mean_pred, log_std_pred

        grad_fn = ops.value_and_grad(
            forward_fn, None, self.optimizer.parameters, has_aux=True
        )

        (loss, mean_pred, log_std_pred), grads = grad_fn()
        grads = self.grad_reducer(grads)
        self.optimizer(grads)

        return loss, mean_pred, log_std_pred

    def validate(self, val_dataset: GeneratorDataset) -> float:
        """Validate model on validation dataset"""
        self.model.set_train(False)
        total_loss = 0.0
        num_batches = 0

        for u, y, target in val_dataset:
            mean_pred, log_std_pred = self.model(u, y)
            loss = self.loss_fn(mean_pred, log_std_pred, target)
            total_loss += float(loss)
            num_batches += 1

        self.model.set_train(True)
        return total_loss / num_batches if num_batches > 0 else float("inf")

    def train(
        self,
        train_dataset: GeneratorDataset,
        val_dataset: Optional[GeneratorDataset] = None,
    ) -> Dict[str, List[float]]:
        """Train the model"""

        epochs = self.training_config["epochs"]
        log_interval = self.training_config.get("log_interval", 10)
        eval_interval = self.training_config.get(
            "eval_interval", 100)  # 100 steps
        verbose = self.training_config.get("verbose", True)

        # Log training data size
        train_data_size = (
            train_dataset.get_dataset_size() * train_dataset.get_batch_size()
        )

        if verbose:
            logger.info(
                "\n***** Probabilistic Training for %d epochs *****\n", epochs)
            logger.info("Train data size: %d", train_data_size)
            steps_per_epoch = train_data_size // train_dataset.get_batch_size()
            logger.info("Steps per epoch: %d", steps_per_epoch)

        # Initialize best values and logger
        best = {}
        best["prob loss"] = float("inf")

        logger_hist = {}
        logger_hist["prob loss"] = []
        logger_hist["val loss"] = []

        global_step = 0

        for epoch in range(epochs):
            self.model.set_train()
            epoch_loss = 0
            batch_count = 0
            for u, y, target in train_dataset:
                step_start_time = time.time()
                loss, _, _ = self.train_step(u, y, target)
                step_time = time.time() - step_start_time
                epoch_loss += float(loss)
                batch_count += 1
                global_step += 1
                # show loss
                if global_step % log_interval == 0:
                    # Negative log likelihood loss can be negative, so we print
                    # its negation for clarity.
                    msg = (
                        f"Epoch {epoch+1}, Step {global_step}, Batch {batch_count}, "
                        f"Loss: {float(loss):.6f}, "
                        f"Step time: {step_time:.3f}s"
                    )
                    logger.info(msg)
                # evaluate
                if val_dataset is not None and global_step % eval_interval == 0:
                    val_loss = self.validate(val_dataset)
                    logger_hist["val loss"].append((global_step, val_loss))
                    # Negative log likelihood loss can be negative
                    msg = f"[Eval] Epoch {epoch+1}, Step {global_step}, Val-Loss: {float(val_loss):.6f} "
                    logger.info(msg)
                    # save best ckpt
                    if val_loss < best["prob loss"]:
                        best["prob loss"] = val_loss
                        self.save_model("best_model.ckpt")

            try:
                avg_epoch_loss = epoch_loss / batch_count
            except ZeroDivisionError as e:
                logger.error(
                    "error: %s, batch size larger than number of training examples", e
                )
                continue
            logger_hist["prob loss"].append(avg_epoch_loss)
            # show after each epoch
            if verbose:
                # Negative log likelihood loss can be negative
                logger.info(
                    "Epoch %d/%d:  Train-Loss: %.6f Best-Loss: %.6f", epoch+1, epoch,
                    float(avg_epoch_loss), float(best['prob loss'])
                )
        return logger_hist

    def save_model(self, filename: str):
        """Save model checkpoint"""
        save_path = os.path.join(self.save_dir, filename)
        ms.save_checkpoint(self.model, save_path)
        logger.info("Model saved to: %s", save_path)

    def load_model(self, filename: str):
        """Load model checkpoint"""
        load_path = os.path.join(self.save_dir, filename)
        if os.path.exists(load_path):
            ms.load_checkpoint(load_path, self.model)
            logger.info("Model loaded from: %s", load_path)
        else:
            logger.warning("Checkpoint not found: %s", load_path)

    def predict(self, u: ms.Tensor, y: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """Make predictions"""
        self.model.set_train(False)
        mean_pred, log_std_pred = self.model(u, y)
        self.model.set_train(True)
        return mean_pred, log_std_pred

    def evaluate(self, test_dataset: GeneratorDataset) -> Dict[str, float]:
        """Evaluate model on test dataset"""
        self.model.set_train(False)

        all_predictions = []
        all_targets = []
        all_means = []
        all_stds = []

        for u, y, target in test_dataset:
            mean_pred, log_std_pred = self.model(u, y)
            std_pred = ops.Exp()(log_std_pred)

            all_predictions.append(mean_pred)
            all_targets.append(target)
            all_means.append(mean_pred)
            all_stds.append(std_pred)

        if all_predictions:
            predictions = ops.Concat(axis=0)(all_predictions)
            targets = ops.Concat(axis=0)(all_targets)
            means = ops.Concat(axis=0)(all_means)
            stds = ops.Concat(axis=0)(all_stds)
        else:
            return {}

        metrics = {}
        metrics["mse"] = compute_mse(targets, predictions)
        metrics["mae"] = compute_mae(targets, predictions)
        metrics["r2"] = compute_r2_score(targets, predictions)

        # Compute relative errors

        calculator = MetricsCalculator()
        metrics["l1_relative_error"] = calculator.l1_relative_error(
            targets, predictions
        )
        metrics["l2_relative_error"] = calculator.l2_relative_error(
            targets, predictions
        )

        # Compute uncertainty quantification metrics
        metrics["fraction_in_ci"] = calculator.fraction_in_ci(
            targets, means, stds, xi=2.0
        )

        # Compute trajectory errors if data is structured as trajectories
        try:
            l1_traj, l2_traj = calculator.trajectory_rel_error(
                targets, predictions)
            metrics["trajectory_l1_error"] = l1_traj
            metrics["trajectory_l2_error"] = l2_traj
        except RuntimeError:
            pass

        return metrics

    def compute_trajectory_metrics(
        self,
        s_test: List[ms.Tensor],
        mean_predictions: List[ms.Tensor],
        std_predictions: List[ms.Tensor],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Compute metrics for trajectory predictions"""
        # Compute L1 and L2 relative errors
        metrics_state = compute_metrics(
            s_test, mean_predictions, ["l1", "l2"], verbose=verbose
        )

        # Compute fraction in confidence interval for each trajectory
        calculator = MetricsCalculator()

        ci_fractions = []
        for i in range(len(s_test)):
            ci_frac = calculator.fraction_in_ci(
                s_test[i], mean_predictions[i], std_predictions[i]
            )
            ci_fractions.append(ci_frac)

        avg_ci_fraction = np.mean(ci_fractions)

        return {
            "l1_metrics": metrics_state[0],  # [max, min, mean]
            "l2_metrics": metrics_state[1],  # [max, min, mean]
            "avg_ci_fraction": avg_ci_fraction,
            "ci_fractions": ci_fractions,
        }


def create_trainer(
    model: nn.Cell,
    config: Dict[str, Any],
    save_dir: str = "outputs",
    distributed: int = 0,
) -> DeepONetTrainer:
    """Create trainer instance"""
    return DeepONetTrainer(model, config, save_dir, distributed)
