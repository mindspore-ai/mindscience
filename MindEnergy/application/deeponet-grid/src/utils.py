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
Utility functions for DeepONet-Grid-UQ
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import yaml
import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np


def _ensure_dir(path: str) -> None:
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


def plot_pred_uq(
    sensors, u, y, s, s_mean, s_std, xlabel="$y$", ylabel="$s^\dagger(u)(y)$",
    size=10, save_path=None, metrics_text=None,
):
    """
    plot prediction with confidence interval
    """
    u = u.reshape(
        -1,
    )
    s = s.reshape(
        -1,
    )
    s_mean = s_mean.reshape(
        -1,
    )
    s_std = s_std.reshape(
        -1,
    )

    plt.figure()
    plt.plot(sensors, u, ":k", label="Input")
    plt.plot(y, s, "-b", label="True")
    plt.plot(y, s_mean, "--r", label="Mean prediction")
    plt.fill(
        np.concatenate([y, y[::-1]]),
        np.concatenate(
            [s_mean - 1.9600 * s_std, (s_mean + 1.9600 * s_std)[::-1]]),
        alpha=0.5,
        fc="c",
        ec="None",
        label="95% confidence interval",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim((0.2, 1.2))
    plt.legend(prop={"size": size})

    if metrics_text:
        ax = plt.gca()
        ax.text(
            0.99,
            0.01,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight", dpi=160)
        plt.close()
    else:
        plt.show()


def parse_loss_log(log_file_path: str) -> Tuple[List[float], List[int], List[int]]:
    """
    Parse loss log file and extract loss values, epochs, and steps
    """
    losses = []
    epochs = []
    steps = []

    if not os.path.exists(log_file_path):
        print(f"Log file not found: {log_file_path}")
        return losses, epochs, steps

    # Regular expression to match the log format
    pattern = r"INFO:root:Epoch (\d+), Step (\d+), Batch \d+, Loss: ([-\d.]+), Step time: \d+\.\d+s"

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = float(match.group(3))

                epochs.append(epoch)
                steps.append(step)
                losses.append(loss)

    print(f"Parsed {len(losses)} loss entries from log file")
    return losses, epochs, steps


def parse_val_loss_log(log_file_path: str) -> Tuple[List[float], List[int], List[int]]:
    """
    Parse validation loss log file and extract validation loss values
    """
    val_losses = []
    epochs = []
    steps = []

    if not os.path.exists(log_file_path):
        print(f"Log file not found: {log_file_path}")
        return val_losses, epochs, steps

    # Regular expression to match validation loss log format
    pattern = r"INFO:root:\[Eval\] Epoch (\d+), Step (\d+), Val-Loss: ([-\d.]+) \(negated for display\)"

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                val_loss = float(match.group(3))

                epochs.append(epoch)
                steps.append(step)
                val_losses.append(val_loss)

    print(f"Parsed {len(val_losses)} validation loss entries from log file")
    return val_losses, epochs, steps


def plot_loss_curves(
    log_file_path: str,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot training and validation loss curves from log file
    """
    losses, epochs, steps = parse_loss_log(log_file_path)
    val_losses, _, val_steps = parse_val_loss_log(log_file_path)

    if not losses:
        print("No loss data found in log file")
        return

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(steps, losses, "b-", label="Training Loss", linewidth=1, alpha=0.8)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss (negated for display)")
    ax1.set_title("Training Loss Curve")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot validation loss if available
    if val_losses:
        ax1.plot(
            val_steps, val_losses, "r-", label="Validation Loss", linewidth=1, alpha=0.8
        )
        ax1.legend()

    # Plot loss vs epochs
    unique_epochs = list(set(epochs))
    epoch_losses = []
    for epoch in unique_epochs:
        epoch_indices = [i for i, e in enumerate(epochs) if e == epoch]
        epoch_loss = np.mean([losses[i] for i in epoch_indices])
        epoch_losses.append(epoch_loss)

    ax2.plot(unique_epochs, epoch_losses, "g-",
             label="Average Epoch Loss", linewidth=2)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Average Loss (negated for display)")
    ax2.set_title("Average Loss per Epoch")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss curve saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def plot_loss_statistics(
    log_file_path: str,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Plot detailed loss statistics including histograms and moving averages
    """
    # Parse loss data
    losses, epochs, steps = parse_loss_log(log_file_path)
    val_losses, _, val_steps = parse_val_loss_log(log_file_path)
    if not losses:
        print("No loss data found in log file")
        return
    # Create figure with subplots
    _ = plt.figure(figsize=figsize)
    # 1. Training loss over steps
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(steps, losses, "b-", alpha=0.7, linewidth=0.8)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss (negated for display)")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    # 2. Moving average of training loss
    ax2 = plt.subplot(2, 3, 2)
    window_size = min(50, len(losses) // 10)  # Adaptive window size
    if window_size > 1:
        moving_avg = np.convolve(
            losses, np.ones(window_size) / window_size, mode="valid"
        )
        steps_avg = steps[window_size - 1:]
        ax2.plot(steps_avg, moving_avg, "r-", linewidth=1.5)
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("Moving Average Loss")
        ax2.set_title(f"Moving Average (window={window_size})")
        ax2.grid(True, alpha=0.3)
    # 3. Loss histogram
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(losses, bins=30, alpha=0.7, color="blue", edgecolor="black")
    ax3.set_xlabel("Loss Value")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Loss Distribution")
    ax3.grid(True, alpha=0.3)
    # 4. Loss vs epochs
    ax4 = plt.subplot(2, 3, 4)
    unique_epochs = list(set(epochs))
    epoch_losses = []
    for epoch in unique_epochs:
        epoch_indices = [i for i, e in enumerate(epochs) if e == epoch]
        epoch_loss = np.mean([losses[i] for i in epoch_indices])
        epoch_losses.append(epoch_loss)
    ax4.plot(unique_epochs, epoch_losses, "g-", linewidth=2, marker="o")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Average Loss")
    ax4.set_title("Average Loss per Epoch")
    ax4.grid(True, alpha=0.3)
    # 5. Validation loss (if available)
    ax5 = plt.subplot(2, 3, 5)
    if val_losses:
        ax5.plot(
            val_steps, val_losses, "orange", linewidth=1.5, label="Validation Loss"
        )
        ax5.set_xlabel("Training Steps")
        ax5.set_ylabel("Validation Loss")
        ax5.set_title("Validation Loss")
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    else:
        ax5.text(
            0.5, 0.5, "No validation loss data",
            ha="center", va="center", transform=ax5.transAxes,
        )
        ax5.set_title("Validation Loss (No Data)")
    # 6. Loss statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    stats_text = (
        f"Loss Statistics:\n----------------\n"
        f"Total Steps: {len(losses)}\n"
        f"Total Epochs: {max(epochs) if epochs else 0}\n"
        f"Min Loss: {min(losses):.6f}\n"
        f"Max Loss: {max(losses):.6f}\n"
        f"Mean Loss: {np.mean(losses):.6f}\n"
        f"Std Loss: {np.std(losses):.6f}\n"
        f"Final Loss: {losses[-1]:.6f}\n"
    )
    if val_losses:
        stats_text += (
            f"\nValidation Stats:\n----------------\n"
            f"Val Steps: {len(val_losses)}\n"
            f"Min Val Loss: {min(val_losses):.6f}\n"
            f"Max Val Loss: {max(val_losses):.6f}\n"
            f"Mean Val Loss: {np.mean(val_losses):.6f}\n"
            f"Final Val Loss: {val_losses[-1]:.6f}\n"
        )

    ax6.text(
        0.05, 0.95, stats_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
    )
    plt.tight_layout()
    # Save plot if specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss statistics saved to: {save_path}")
    # Show plot if requested
    if show_plot: plt.show()
    plt.close()


def load_training_history(history_file_path: str) -> Dict:
    """
    Load training history from JSON file
    """
    if not os.path.exists(history_file_path):
        print(f"History file not found: {history_file_path}")
        return {}

    with open(history_file_path, "r") as f:
        history = json.load(f)

    return history


def plot_training_history(
    history_file_path: str, save_path: Optional[str] = None, show_plot: bool = True
) -> None:
    """Plot training history from JSON file"""
    history = load_training_history(history_file_path)

    if not history:
        print("No training history data found")
        return

    _, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    if len(history) == 1:
        axes = [axes]

    for i, (key, values) in enumerate(history.items()):
        axes[i].plot(values, "b-", linewidth=1.5)
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(key.replace("_", " ").title())
        axes[i].set_title(f'{key.replace("_", " ").title()} vs Epochs')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    plt.close()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(model, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ms.load_checkpoint(checkpoint_path, model)
    return model


def extract_log(log_file: str, history_file: str):
    """
    Extract Log.
    """
    print("DeepONet-Grid-UQ training log extraction")
    print("=" * 40)

    if os.path.exists(log_file):
        print(f"Found log file: {log_file}")

        # Parse and print basic statistics
        losses, _, _ = parse_loss_log(log_file)
        if losses:
            print(f"Loss statistics:")
            print(f"  Total entries: {len(losses)}")
            print(f"  Min loss: {min(losses):.6f}")
            print(f"  Max loss: {max(losses):.6f}")
            print(f"  Mean loss: {np.mean(losses):.6f}")
            print(f"  Final loss: {losses[-1]:.6f}")

        # Plot loss curves
        plot_loss_curves(log_file, save_path="loss_curves.png")
        plot_loss_statistics(log_file, save_path="loss_statistics.png")

    # Check for training history
    if os.path.exists(history_file):
        print(f"Found training history: {history_file}")
        plot_training_history(history_file, save_path="training_history.png")

    print("Completed!")
