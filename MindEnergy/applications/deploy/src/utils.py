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
Utilities module for MindScience deployment service.

This module provides a collection of utility functions and classes for various
tasks in the MindScience deployment service, including:

- File handling operations (upload, extraction, HDF5 read/write)
- Model file collection and management
- Input data extension for batch processing
- Task status counting
- NPU (Neural Processing Unit) resource monitoring
- Asynchronous operations support

The Utilities class contains static methods that are commonly used across
different components of the deployment service.
"""

import os
import re
import zipfile
import tarfile
import asyncio
import traceback
import subprocess
from typing import Dict, List

import h5py
import aiofiles
import numpy as np
from loguru import logger
from fastapi import UploadFile, HTTPException

from .enums import TaskStatus
from .config import DeployConfig, ModelConfig

class Utilities:
    """
    Utilities helper class that encapsulates deployment-related utility functions.

    This class includes file handling, model collection, HDF5 file read/write,
    asynchronous file upload saving, and methods to collect NPU usage information.
    """
    @staticmethod
    def collect_mindir_models(root_dir: str) -> Dict[str, List[str]]:
        """Collect *.mindir model files from each subdirectory under the specified root directory.

        Args:
            root_dir (str): Root directory path to search. Each immediate subdirectory
                is treated as a distinct model name.

        Returns:
            Dict[str, List[str]]: Mapping of model directory name to a sorted list of
                absolute paths of all `.mindir` files found in that model directory.

        Raises:
            ValueError: If `root_dir` is not a valid directory.
        """
        if not os.path.isdir(root_dir):
            raise ValueError(f"Invalid directory: {root_dir}")

        model_dict = {}
        for model_name in os.listdir(root_dir):
            model_dir = os.path.join(root_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            mindir_files = []
            for file in os.listdir(model_dir):
                if file.endswith(".mindir"):
                    abs_path = os.path.abspath(os.path.join(model_dir, file))
                    mindir_files.append(abs_path)

            if mindir_files:
                model_dict[model_name] = sorted(mindir_files)

        return model_dict

    @staticmethod
    def extend_input(array: np.ndarray, batch_size: int) -> np.ndarray:
        """Extend (pad) the input array along the first dimension so that its length
        is divisible by `batch_size`.

        If the first dimension is already divisible by `batch_size`, the original
        array is returned. Otherwise, zero-rows are appended to the end of the
        array to make the first dimension divisible.

        Args:
            array (np.ndarray): The numpy array to pad. Must have at least one dimension.
            batch_size (int): The batch size to make the first dimension divisible by.

        Returns:
            np.ndarray: The padded numpy array with the same dtype as the input.

        Raises:
            ValueError: If `array` has no shape (empty).
        """
        if not array.shape:
            raise ValueError(f"The array to extend is empty, please check: {array}.")

        if array.shape[0] % batch_size != 0:
            padding_length = batch_size - array.shape[0] % batch_size
            padding_contents = np.zeros((padding_length, *array.shape[1:]), dtype=array.dtype)
            return np.concatenate([array, padding_contents], axis=0)
        return array

    @staticmethod
    async def save_upload_file(upload_file: UploadFile, destination: str):
        """Asynchronously save a FastAPI `UploadFile` to the specified destination
        path in chunks.

        Args:
            upload_file (UploadFile): The uploaded file object provided by FastAPI.
            destination (str): Target file path (including filename) to save the upload.

        Returns:
            None

        Raises:
            HTTPException: Raises an HTTPException (status 500) if saving fails.
        """
        try:
            async with aiofiles.open(destination, "wb") as f:
                while chunk := await upload_file.read(DeployConfig.chunk_size):
                    await f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"save upload file failed: {e}") from e

    @staticmethod
    def extract_file(file_path: str, extract_to: str = None):
        """Extract supported archive file formats (tar, tar.gz, tgz, tar.bz2, tbz2,
        tar.xz, txz, zip) to a target directory.

        Args:
            file_path (str): Path to the archive file to extract.
            extract_to (str, optional): Target directory to extract into. If None,
                the archive's containing directory is used.

        Returns:
            None

        Raises:
            FileNotFoundError: If `file_path` does not exist.
            ValueError: If the file format is not a supported archive type.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if extract_to is None:
            extract_to = os.path.dirname(file_path)

        if file_path.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
            with tarfile.open(file_path, 'r:*') as tf:
                tf.extractall(extract_to)
        elif file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported compressed file format: {file_path}.")

    @staticmethod
    def count_pending_task(tasks_status: Dict[str, str]) -> int:
        """Count the number of tasks in a mapping that are in the PENDING state.

        Args:
            tasks_status (Dict[str, str]): Mapping from task ID to task status
                (expected to be members of `TaskStatus`).

        Returns:
            int: Number of tasks whose status equals `TaskStatus.PENDING`.
        """
        pending_number = 0
        for value in tasks_status.values():
            if value == TaskStatus.PENDING:
                pending_number += 1
        return pending_number

    @staticmethod
    def load_h5_file(file_path: str) -> List[np.ndarray]:
        """Load input columns specified in `ModelConfig.input_columns` from an HDF5
        file and return them as a list of numpy arrays.

        Args:
            file_path (str): HDF5 file path to read from.

        Returns:
            List[np.ndarray]: List of numpy arrays for each input column defined in
                `ModelConfig.input_columns`, preserving the same order.

        Raises:
            ValueError: If any key from `ModelConfig.input_columns` is missing in the file.
        """
        batch_inputs = []
        with h5py.File(file_path, "r") as f:
            for key in ModelConfig.input_columns:
                if key not in f.keys():
                    raise ValueError(f"Key {key} not in dataset, please check!")
                batch_inputs.append(np.array(f[key], dtype=f[key].dtype))
        return batch_inputs

    @staticmethod
    def save_h5_file(items: List[np.ndarray], file_path: str):
        """Write prediction outputs to an HDF5 file.

        Output column names are taken from `ModelConfig.output_columns`. If the
        number of `items` does not match the number of configured output column
        names, the function will rename columns using the default format
        `output_0`, `output_1`, ... and log a warning.

        Args:
            items (List[np.ndarray]): List of numpy arrays to write as outputs.
            file_path (str): HDF5 output file path.

        Returns:
            None

        Notes:
            If the number of configured output columns does not match the number
            of provided items, the method assigns default names and logs a
            warning.
        """
        output_columns = ModelConfig.output_columns
        if len(items) != len(output_columns):
            logger.warning("Number of outputs in config is inconsistent with the number of the actual outputs, \
                           rename the output column.")
            output_columns = tuple(f"output_{i}" for i in  range(len(items)))

        with h5py.File(file_path, "w") as f:
            for key, value in zip(output_columns, items):
                f.create_dataset(key, data=value)

    @staticmethod
    async def get_npu_usage():
        """Asynchronously execute `npu-smi info` to obtain NPU utilization and memory
        information, parse the output, and return a structured list of stats.

        Each returned item is a dictionary with the following keys:
            - id (int): NPU device ID
            - utilization_percent (int): NPU utilization percentage
            - memory_used_mb (int): Memory used in MB
            - memory_total_mb (int): Total memory in MB

        Returns:
            List[Dict[str, int]]: List of dictionary entries containing NPU stats.

        Raises:
            FileNotFoundError: If the `npu-smi` command is not available.
            RuntimeError: If executing `npu-smi info` fails for other reasons.
        """
        try:
            command = ["npu-smi", "info"]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            output_bytes, stderr_bytes = await process.communicate()

            if process.returncode != 0:
                if b"command not found" in stderr_bytes.lower() or \
                   b"no such file or directory" in stderr_bytes.lower():
                    logger.error("ERROR: 'npu-smi' command not found!")
                    logger.error("Please ensure that the Ascend CANN toolkit is correctly installed and \
                          its bin directory has been added to the system's PATH environment variable.")
                    raise FileNotFoundError("'npu-smi' command not found.")
                raise subprocess.CalledProcessError(
                    process.returncode, command, output=output_bytes, stderr=stderr_bytes
                )
        except Exception as e:
            logger.error(f"Execute 'npu-smi info' failed, ERROR: {e}")
            raise RuntimeError(f"Execute 'npu-smi info' failed, ERROR: {e}") from e

        logger.info("=== npu-smi info original output ===")
        logger.info(output_bytes.decode("utf-8", errors="replace"))
        logger.info("===========================")

        output = output_bytes.decode("utf-8", errors="ignore")

        npu_stats = []
        lines = output.strip().splitlines()

        if not lines:
            logger.warning("'npu-smi info' did not return any output.")
            return npu_stats

        for i, line in enumerate(lines):
            match_first_line = re.match(r"^\s*\|\s*(\d+)\s+[a-zA-Z0-9]+\s*\|", line)

            if not match_first_line:
                logger.warning(f"Can not match NPU ID in line: {line}.")
                continue

            try:
                npu_id_str = match_first_line.group(1)
                if not npu_id_str:
                    logger.warning(f"NPU ID is empty in line: {line}.")
                    continue

                npu_id = int(npu_id_str)
                logger.info(f"Found NPU ID: {npu_id}.")

                if (i + 1) >= len(lines):
                    logger.warning("Not enough rows to parse NPU utilization information.")
                    continue

                second_line = lines[i + 1]
                logger.info(f"Process NPU {npu_id} data line: {second_line}.")

                parts = second_line.split("|")
                if len(parts) < 4:
                    logger.warning(f"The data row format is incorrect and \
                                   the number of columns is insufficient: {second_line}.")
                    continue

                data_string = parts[-2].strip()
                logger.info(f"Extracted data: '{data_string}'.")

                tokens = re.findall(r"\d+", data_string)
                logger.info(f"Extracted tokens: {tokens}.")

                if len(tokens) < 5:
                    logger.warning(f"Insufficient tokens: {tokens}.")
                    continue

                try:
                    utilization_percent = int(tokens[0])
                    mem_used = int(tokens[3])
                    mem_total = int(tokens[4])

                    info = {
                        "id": npu_id,
                        "utilization_percent": utilization_percent,
                        "memory_used_mb": mem_used,
                        "memory_total_mb": mem_total,
                    }
                    logger.info(f"Successfully parse {npu_id} info: {info}.")
                    npu_stats.append(info)

                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing NPU {npu_id} data: {e}.")
                    continue

            except Exception as e:
                logger.exception(f"Unexpected error when parsing NPU data: {e}.")
                traceback.print_exc()
                continue

        if not npu_stats:
            logger.warning(f"Can not parse any info from 'npu-smi info' output. original output: \n{output}")
            if "No NPU device found" in output:
                logger.info("'No NPU device found' - Perhaps there are no available NPU devices.")
        else:
            logger.info(f"Successfully parse {len(npu_stats)} NPU info.")

        return npu_stats
