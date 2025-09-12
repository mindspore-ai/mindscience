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
"""Data processing for deeponet-grid"""

import os
from typing import Dict, Tuple

import numpy as np
import mindspore as ms
from mindspore.communication import get_group_size, get_rank
from mindspore.dataset import GeneratorDataset
from mindspore.ops import operations as ops


class DataGenerator:
    """Data generator for MindSpore training"""

    def __init__(
        self, u: np.ndarray, y: np.ndarray, s: np.ndarray, dtype: str = "float32"
    ):
        self.u = u
        self.y = y
        self.s = s
        self.len = len(u)
        self.dtype = ms.float32 if dtype == "float32" else ms.float64

    def __getitem__(self, index):
        u_data = self.u[index]
        y_data = self.y[index]
        s_data = self.s[index]

        if len(y_data.shape) != 1:
            raise ValueError(f"y_data must be 1D, got shape {y_data.shape}")

        if len(s_data.shape) != 1:
            raise ValueError(f"s_data must be 1D, got shape {s_data.shape}")

        u_tensor = ms.Tensor(u_data, self.dtype)
        y_tensor = ms.Tensor(y_data, self.dtype)
        s_tensor = ms.Tensor(s_data, self.dtype)

        return u_tensor, y_tensor, s_tensor

    def __len__(self):
        return self.len


def generate_synthetic_data(
    n_samples: int = 1000, n_sensors: int = 33, n_points: int = 1, seed: int = 1234
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for testing"""
    np.random.seed(seed)

    # Generate input functions (u) - random functions
    u = np.random.randn(n_samples, n_sensors)

    # Generate evaluation points (y)
    y = np.random.rand(n_samples, n_points) * 2.0  # Random points in [0, 2]

    # Generate true solutions (s) - simple example using sin function
    s = np.sin(u.mean(axis=1, keepdims=True) * y)

    # Prepare data for DeepONet (expand to single query points)
    expanded_u, expanded_y, expanded_s, _ = prepare_deeponet_data(u, y, s)
    return expanded_u, expanded_y, expanded_s


def load_real_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real data from npz file

    Args:
        data_path: Path to the npz file

    Returns:
        Tuple of (u, y, s) where:
        - u: Input function values at sensor locations
        - y: Evaluation points
        - s: True solution values
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = np.load(data_path)

    # Check for required fields
    required_fields = ["u", "y", "s"]
    for field in required_fields:
        if field not in data.files:
            raise ValueError(f"Required field '{field}' not found in data file")

    return data["u"], data["y"], data["s"]


def normalize_data(
    u: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    method: str = "standard",
    dtype: str = "float32",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Normalize data

    Args:
        u: Input function values
        y: Evaluation points
        s: Solution values
        method: Normalization method ('standard', 'minmax', 'none')

    Returns:
        Tuple of (u_norm, y_norm, s_norm, scalers)
    """
    scalers = {}

    if method == "none":
        return u, y, s, scalers

    if dtype == "float32":
        dtype = ms.float32
    elif dtype == "float64":
        dtype = ms.float64

    # Convert to tensors for computation
    u_tensor = ms.Tensor(u, dtype)
    y_tensor = ms.Tensor(y, dtype)
    s_tensor = ms.Tensor(s, dtype)

    # Operations
    reduce_mean = ops.ReduceMean(keep_dims=True)
    reduce_std = ops.ReduceStd(keep_dims=True)
    reduce_min = ops.ReduceMin(keep_dims=True)
    reduce_max = ops.ReduceMax(keep_dims=True)

    if method == "standard":
        # Standard normalization: (x - mean) / std
        u_mean = reduce_mean(u_tensor, 0)
        u_std = reduce_std(u_tensor)[0]
        u_norm = (u_tensor - u_mean) / (u_std[0] + 1e-8)
        y_mean = reduce_mean(y_tensor, 0)
        y_std = reduce_std(y_tensor)
        y_norm = (y_tensor - y_mean) / (y_std[0] + 1e-8)
        s_mean = reduce_mean(s_tensor, 0)
        s_std = reduce_std(s_tensor)
        s_norm = (s_tensor - s_mean) / (s_std[0] + 1e-8)
        # Store scalers for later use
        scalers["u"] = {"mean": u_mean.asnumpy(), "std[0]": u_std[0].asnumpy()}
        scalers["y"] = {"mean": y_mean.asnumpy(), "std[0]": y_std[0].asnumpy()}
        scalers["s"] = {"mean": s_mean.asnumpy(), "std[0]": s_std[0].asnumpy()}

    elif method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        u_min = reduce_min(u_tensor, 0)
        u_max = reduce_max(u_tensor, 0)
        u_norm = (u_tensor - u_min) / (u_max - u_min + 1e-8)

        y_min = reduce_min(y_tensor, 0)
        y_max = reduce_max(y_tensor, 0)
        y_norm = (y_tensor - y_min) / (y_max - y_min + 1e-8)

        s_min = reduce_min(s_tensor, 0)
        s_max = reduce_max(s_tensor, 0)
        s_norm = (s_tensor - s_min) / (s_max - s_min + 1e-8)

        # Store scalers for later use
        scalers["u"] = {"min": u_min.asnumpy(), "max": u_max.asnumpy()}
        scalers["y"] = {"min": y_min.asnumpy(), "max": y_max.asnumpy()}
        scalers["s"] = {"min": s_min.asnumpy(), "max": s_max.asnumpy()}

    # Convert back to numpy
    u_norm = u_norm.asnumpy()
    y_norm = y_norm.asnumpy()
    s_norm = s_norm.asnumpy()

    return u_norm, y_norm, s_norm, scalers


def split_data(
    u: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Tuple]:
    """Split data into train, validation, and test sets

    Args:
        u: Input function values
        y: Evaluation points
        s: Solution values
        train_ratio: Training set fraction
        val_ratio: Validation set fraction
        test_ratio: Test set fraction
        random_state: Random seed

    Returns:
        Dictionary containing train, validation, and test data
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Splits must sum to 1.0"

    # Set random seed
    np.random.seed(random_state)

    n_samples = len(u)
    indices = np.random.permutation(n_samples)

    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    data_splits = {
        "train": (u[train_indices], y[train_indices], s[train_indices]),
        "val": (u[val_indices], y[val_indices], s[val_indices]),
        "test": (u[test_indices], y[test_indices], s[test_indices]),
    }

    return data_splits


def create_datasets(
    data_splits: Dict[str, Tuple], batch_size: int = 32, distributed: int = 0
) -> Dict[str, GeneratorDataset]:
    """Create datasets from data splits

    Args:
        data_splits: Dictionary containing train, val, test data
        batch_size: Batch size for training

    Returns:
        Dictionary containing datasets
    """
    datasets = {}

    for split_name, (u, y, s) in data_splits.items():
        # Create data generator
        data_gen = DataGenerator(u, y, s)

        # Create dataset with three columns: u, y, s
        # ====== Data Parallel: add num_shards and shard_id for train set =====
        if split_name == "train":
            if distributed:
                try:
                    rank_id = get_rank()
                    rank_size = get_group_size()
                except RuntimeError:
                    rank_id = 0
                    rank_size = 1
                dataset = GeneratorDataset(
                    source=data_gen,
                    column_names=["u", "y", "s"],
                    shuffle=True,
                    num_shards=rank_size,
                    shard_id=rank_id,
                ).batch(batch_size)
            else:
                dataset = GeneratorDataset(
                    source=data_gen, column_names=["u", "y", "s"], shuffle=True
                ).batch(batch_size)
        else:
            dataset = GeneratorDataset(
                source=data_gen, column_names=["u", "y", "s"], shuffle=False
            ).batch(batch_size)
        # ====== End Data Parallel ======

        # set batch size
        datasets[split_name] = dataset

    return datasets


def prepare_deeponet_data(u, y, s):
    """
    Prepare data for DeepONet training from user's data format
    """

    n_samples, n_sensors = u.shape
    y_n_samples, y_n_points = y.shape
    s_n_samples, s_n_points = s.shape

    if y_n_samples != s_n_samples or n_samples != y_n_samples:
        raise ValueError(
            f"u, y and s must have the same number of samples, got {n_samples}, {y_n_samples} and {s_n_samples}"
        )

    if y_n_points != s_n_points:
        raise ValueError(
            f"y and s must have the same number of points, got {y_n_points} and {s_n_points}"
        )

    n_points = y_n_points

    # expand time points to batch dimension
    # from (n_samples, n_sensors) and (n_samples, n_points)
    # to (n_samples * n_points, n_sensors) and (n_samples * n_points, 1)

    # repeat u_batch to match each time point

    # Expand data for DeepONet training
    u_expanded = []
    y_expanded = []
    s_expanded = []

    # np.tile + reshape is not applicable for large data
    for i in range(n_samples):
        for j in range(n_points):
            u_expanded.append(u[i])  # Same input function for all time points
            y_expanded.append([y[i][j]])  # Single time point
            s_expanded.append([s[i, j]])  # Target value at this time point

    u_expanded = np.array(u_expanded)
    y_expanded = np.array(y_expanded)
    s_expanded = np.array(s_expanded)

    metadata = {
        "n_samples": n_samples,
        "n_sensors": n_sensors,
        "n_points": n_points,
        "original_shapes": {"u": u.shape, "y": y.shape, "s": s.shape},
        "expanded_shapes": {
            "u": u_expanded.shape,
            "y": y_expanded.shape,
            "s": s_expanded.shape,
        },
    }
    return u_expanded, y_expanded, s_expanded, metadata


def trajectory_prediction(u: np.ndarray, time_points: np.ndarray, model) -> np.ndarray:
    """
    Predict trajectory for a single sample using DeepONet model
    """

    # Ensure u is 2D for model input
    if len(u.shape) == 1:
        u = u.reshape(1, -1)  # (1, n_sensors)
    n_time_points = len(time_points)

    predictions = []

    # create query points for each time point of the sample
    u_expanded = np.repeat(u, n_time_points, axis=0)
    y_expanded = np.tile(
        time_points.reshape(-1, 1), (1, 1)
    )  # (batch_size * n_time_points, 1)

    u_tensor = ms.Tensor(u_expanded, ms.float32)
    y_tensor = ms.Tensor(y_expanded, ms.float32)

    # one inference to get all predictions
    mean_pred, log_std_pred = model(u_tensor, y_tensor)

    # reshape results
    predictions = mean_pred.reshape(1, n_time_points, 1)
    std_predictions = ms.ops.exp(log_std_pred).reshape(1, n_time_points, 1)

    return np.array(predictions).reshape(-1, 1), np.array(std_predictions).reshape(
        -1, 1
    )


def batch_trajectory_prediction(model, u_batch, time_points):
    """
    Batch prediction of trajectories,
    this function is used for small test set,
    dont input too much samples in one batch.
    """

    batch_size = u_batch.shape[0]
    n_time_points = len(time_points)

    # expand time points to batch dimension
    u_expanded = np.repeat(
        u_batch, n_time_points, axis=0
    )  # (batch_size * n_time_points, n_sensors)

    y_expanded = np.tile(
        time_points.reshape(-1, 1), (batch_size, 1)
    )  # (batch_size * n_time_points, 1)

    u_tensor = ms.Tensor(u_expanded, ms.float32)
    y_tensor = ms.Tensor(y_expanded, ms.float32)

    mean_pred, log_std_pred = model(u_tensor, y_tensor)

    mean_predictions = mean_pred.reshape(batch_size, n_time_points, 1)
    std_predictions = ms.ops.exp(log_std_pred).reshape(batch_size, n_time_points, 1)

    return mean_predictions, std_predictions


def load_and_preprocess_real_data(config: dict):
    """
    Load and split data, then create datasets for training and inference.
    """
    data_path = config["data"]["data_path"]

    u, y, s = load_real_data(data_path)

    u_expanded, y_expanded, s_expanded, prep_metadata = prepare_deeponet_data(u, y, s)

    data_splits = split_data(
        u_expanded,
        y_expanded,
        s_expanded,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
    )

    datasets = create_datasets(
        data_splits,
        batch_size=config["training"]["batch_size"],
        distributed=(
            config["training"]["distributed"]
            if config["training"].get("distributed") == 1
            else 0
        ),
    )

    metadata = {
        "input_dim": u_expanded.shape[-1],
        "output_dim": s_expanded.shape[-1],
        "n_samples": len(u_expanded),
        "n_sensors": u.shape[-1],  # Original sensor count
        "data_shapes": {
            "u": u_expanded.shape,
            "y": y_expanded.shape,
            "s": s_expanded.shape,
        },
    }

    if prep_metadata:
        metadata.update(prep_metadata)

    return datasets, metadata
