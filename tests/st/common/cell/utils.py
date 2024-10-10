# ============================================================================
# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Utils"""
import os
import logging

import numpy as np


logger = logging.getLogger("__name__")


def check_path_exists(path):
    return os.path.exists(path)


def compare_output(output_1, output_2, rtol=1e-5, atol=1e-5):
    r"""
    Compares model outputs and determines if they match within the specified tolerance
    Args:
        output_1 (Union[np.ndarray, Tuple[np.ndarray, ...]]): First model output to compare
        output_2 (Union[np.ndarray, Tuple[np.ndarray, ...]]): Second model output to compare
        rtol (float): Relative tolerance for allowed error, default is 1e-5
        atol (float): Absolute tolerance for allowed error, default is 1e-5

    Returns:
        bool: Whether the outputs match within the given tolerance
    """
    # Output of tensor
    if isinstance(output_1, np.ndarray):
        return np.allclose(output_1, output_2, rtol, atol, equal_nan=True)
    # Output of tuple of tensors
    if isinstance(output_1, tuple):
        # Loop through tuple of outputs
        for i, (out_1, out_2) in enumerate(zip(output_1, output_2)):
            # If tensor use allclose
            if isinstance(out_1, np.ndarray):
                if not np.allclose(out_1, out_2, rtol, atol, equal_nan=True):
                    logger.warning("Failed comparison between outputs %d", i)
                    logger.warning("Max Difference: %f", np.max(np.abs(out_1 - out_2)))
                    logger.warning("Difference: %s", (out_1 - out_2))
                    return False
            # Otherwise assume primitive
            else:
                if not out_1 == out_2:
                    return False
    # Unsupported output type
    else:
        logger.error(
            "Model returned invalid type for unit test, should be np.ndarray or Tuple[np.ndarray]"
        )
        return False

    return True
