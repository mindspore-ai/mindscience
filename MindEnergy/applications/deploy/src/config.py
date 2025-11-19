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

"""
Configuration module for MindScience deployment service.

This module defines configuration classes and functions used throughout the
MindScience deployment and monitoring services. It includes:

- ModelConfig: Specifies model input and output column names
- DeployConfig: Contains deployment-related settings such as device numbers, request limits, and file paths
- ServerConfig: Defines server settings including host, ports, and connection parameters
- configure_logging: Function to set up logging for different services

The configurations use dataclasses with frozen=True to ensure immutability
and thread-safe access to configuration values.
"""

import sys
from typing import Tuple
from dataclasses import dataclass

from loguru import logger

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model input and output specifications.

    Attributes:
        input_columns: Tuple of column names used as model inputs.
        output_columns: Tuple of column names produced as model outputs.
    """
    input_columns: Tuple[str] = ("x", "edge_index", "edge_attr")

    output_columns: Tuple[str] = ("output",)


@dataclass(frozen=True)
class DeployConfig:
    """Configuration for deployment settings.

    Attributes:
        max_device_num: Maximum number of devices allowed for deployment.
        deploy_device_num: Number of devices to be used for deployment.
        max_request_num: Maximum number of concurrent requests allowed.
        models_dir: Directory path for storing model files.
        datasets_dir: Directory path for storing dataset files.
        results_dir: Directory path for storing result files.
        dummy_model_path: File path for the dummy model used in testing.
        chunk_size: Size of data chunks for processing in bytes.
    """
    max_device_num: int = 8

    deploy_device_num: int = 8

    max_request_num: int = 100

    models_dir: str = "models"

    datasets_dir: str = "datasets"

    results_dir: str = "results"

    dummy_model_path: str = "dummy_model.mindir"

    chunk_size: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for server settings.

    Attributes:
        host: Host address for the server.
        deploy_port: Port number for deployment service.
        monitor_port: Port number for monitoring service.
        limit_concurrency: Maximum number of concurrent connections allowed.
        timeout_keep_alive: Timeout duration for keep-alive connections in seconds.
        backlog: Maximum number of pending connections in the queue.
    """
    host: str = "127.0.0.1"

    deploy_port: int = 8001

    monitor_port: int = 8002

    limit_concurrency: int = 1000

    timeout_keep_alive: int = 30

    backlog: int = 2048


def configure_logging(service: str):
    """Configures logging settings for the specified service.

    Args:
        service: Name of the service to configure logging for.
    """
    logger.add(
        f"logs/{service}.log",
        rotation="100 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True
    )
    logger.add(sys.stderr, level="DEBUG")
