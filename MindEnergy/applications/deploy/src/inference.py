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
Inference module for MindScience deployment service.

This module provides classes and functions for performing inference using
MindSpore Lite across multiple devices (e.g. NPU devices) in parallel.
It includes:

- InferenceModel: A wrapper for MindSpore Lite models with input/output handling
- MultiprocessInference: A class for managing parallel inference across multiple devices
- infer_process_func: A function that runs in separate processes to handle inference tasks

The module implements a multiprocessing approach where each device runs in its own
process with communication handled via pipes. It provides methods for model building,
inference execution, health checks, and resource cleanup.
"""

import os
from typing import List
from multiprocessing import Process, Pipe

import numpy as np
from loguru import logger

import mindspore_lite as mslite

from .config import DeployConfig
from .enums import ProcessMessage

def infer_process_func(pipe_child_end, model_path, device_id):
    """Function that runs in a separate process to handle inference tasks.

    This function initializes an InferenceModel, handles communication through
    a pipe, and processes different types of messages including initialization,
    inference requests, health checks, and exit signals. It runs in a continuous
    loop until it receives an exit message.

    Args:
        pipe_child_end: The child end of a multiprocessing.Pipe used for
                       communication with the parent process.
        model_path (str): Path to the model file to load for inference.
        device_id (int): ID of the device to run inference on.
    """
    try:
        infer_model = InferenceModel(model_path, device_id)
        if DeployConfig.dummy_model_path.endswith(".mindir") and os.path.exists(DeployConfig.dummy_model_path):
            dummy_model = InferenceModel(DeployConfig.dummy_model_path, device_id)
        else:
            dummy_model = infer_model
        pipe_child_end.send((ProcessMessage.BUILD_FINISH, infer_model.batch_size))
    except Exception as e:
        logger.exception(f"Model initialization failed on device {device_id}: {e}")
        pipe_child_end.send((ProcessMessage.ERROR, str(e)))
        pipe_child_end.close()
        return

    try:
        while True:
            msg_type, msg = pipe_child_end.recv()
            if msg_type == ProcessMessage.EXIT:
                logger.info(f"Device {device_id} received exit signal")
                break
            if msg_type == ProcessMessage.CHECK:
                try:
                    _ = dummy_model.dummy_infer()
                    pipe_child_end.send((ProcessMessage.REPLY, ""))
                except Exception as e:
                    err_msg = f"Dummy inference failed on device {device_id}: {e}"
                    logger.exception(err_msg)
                    pipe_child_end.send((ProcessMessage.ERROR, err_msg))
                    raise RuntimeError(err_msg) from e
            elif msg_type == ProcessMessage.PREDICT:
                try:
                    inputs = msg
                    result = infer_model.infer(inputs)
                    pipe_child_end.send((ProcessMessage.REPLY, result))
                except Exception as e:
                    err_msg = f"Inference failed on device {device_id}: {e}"
                    logger.exception(err_msg)
                    pipe_child_end.send((ProcessMessage.ERROR, err_msg))
                    raise RuntimeError(err_msg) from e
            else:
                err_msg = f"Unexpected message type {msg_type} on device {device_id}"
                logger.exception(err_msg)
                pipe_child_end.send((ProcessMessage.ERROR, err_msg))
                raise RuntimeError(err_msg)
    except Exception as e:
        logger.exception(f"Runtime error on device {device_id}: {e}")
        pipe_child_end.send((ProcessMessage.ERROR, str(e)))
    finally:
        pipe_child_end.close()


class MultiprocessInference:
    """A class for performing inference using multiple processes across different devices.

    This class manages multiple inference processes that run in parallel across specified
    devices (e.g. Ascend NPU devices). It provides methods to build the model, run
    inference, perform health checks, and clean up resources.

    Attributes:
        model_path (str): Path to the model file to be used for inference.
        device_num (int): Number of devices to use for parallel inference.
        batch_size (int): Batch size of the loaded model, initialized to -1.
        process_pool (list): List of Process objects for the inference processes.
        parent_pipes (list): List of parent ends of Pipe objects for communication
                           with the inference processes.
    """

    def __init__(self, model_path: str, device_num: int):
        """Initializes the MultiprocessInference instance.

        Args:
            model_path (str): Path to the model file to be used for inference.
            device_num (int): Number of devices to use for parallel inference.
        """
        self.model_path = model_path
        self.device_num = device_num
        self.batch_size = -1

        self.process_pool = []
        self.parent_pipes = []

    def build_model(self):
        """Builds and initializes the model on multiple devices.

        This method creates a process for each device, initializes the model in each process,
        and establishes communication pipes between the main process and the worker processes.
        It also verifies successful initialization of each device and ensures batch size
        consistency across devices.

        Raises:
            RuntimeError: If model has already been loaded or if initialization fails.
        """
        if self.process_pool:
            self._cleanup_resources()
            raise RuntimeError("The model has been loaded. \
                               Please uninstall this model first and then reload another model!")

        for device_id in range(self.device_num):
            parent_conn, child_conn = Pipe(duplex=True)
            self.parent_pipes.append(parent_conn)

            process = Process(
                target=infer_process_func,
                args=(child_conn, self.model_path, device_id),
                daemon=True,
            )
            self.process_pool.append(process)
            process.start()

            child_conn.close()

        for device_id in range(self.device_num):
            try:
                msg_type, msg = self.parent_pipes[device_id].recv()
                if msg_type == ProcessMessage.ERROR:
                    raise RuntimeError(f"Device {device_id} initialization failed: {msg}")
                if msg_type != ProcessMessage.BUILD_FINISH:
                    raise RuntimeError(f"Unexpected message type {msg_type} during initialization")

                if self.batch_size not in (-1, msg):
                    raise RuntimeError("Batch size in different models are inconsistent.")
                self.batch_size = msg
                logger.info(f"Device {device_id} initialized successfully")
            except Exception as e:
                self._cleanup_resources()
                logger.critical("Failed to initialize inference processes")
                raise RuntimeError(f"Unexpected error {e} during initialization") from e

    def infer(self, inputs):
        """Performs inference on the provided input data.

        This method distributes the input data across available devices and collects
        the results. Each input item is sent to a separate device for parallel processing.

        Args:
            inputs: List of input data for inference. Each item will be sent to a separate device.

        Returns:
            list: List of inference results from each device.

        Raises:
            ValueError: If the number of inputs exceeds the number of available devices.
            RuntimeError: If sending prediction command to a device fails or if inference fails.
        """
        if self.device_num < len(inputs):
            raise ValueError(f"Inputs length {len(inputs)} should be less than or equal to \
                           parallel number {self.device_num}, inference abort!")

        for device_id, input_data in enumerate(inputs):
            try:
                self.parent_pipes[device_id].send((ProcessMessage.PREDICT, input_data))
            except (BrokenPipeError, EOFError, OSError) as e:
                self._cleanup_resources()
                raise RuntimeError(f"Failed to send PREDICT to device {device_id}: {e}") from e
            except Exception as e:
                self._cleanup_resources()
                raise RuntimeError(f"Unexpected error when sending PREDICT to device {device_id}: {e}") from e

        results = []
        for device_id in range(len(inputs)):
            try:
                msg_type, msg = self.parent_pipes[device_id].recv()
                if msg_type == ProcessMessage.ERROR:
                    self._cleanup_resources()
                    raise RuntimeError(f"Device {device_id} inference failed: {msg}, cleanup all resource!")
                results.append(msg)
            except EOFError as e:
                self._cleanup_resources()
                raise RuntimeError(f"Device {device_id} connection closed unexpectedly.") from e

        return results

    def finalize(self):
        """Finalizes the inference processes and cleans up resources.

        This method sends an EXIT signal to all worker processes, waits for them to
        finish, and then cleans up all resources including the processes and pipes.
        """
        for idx, pipe in enumerate(self.parent_pipes):
            try:
                pipe.send((ProcessMessage.EXIT, ""))
            except (BrokenPipeError, EOFError, OSError) as e:
                logger.exception(f"Failed to send EXIT to device {idx}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error when sending EXIT to device {idx}: {e}")

        for idx, process in enumerate(self.process_pool):
            if process.is_alive():
                try:
                    process.join(timeout=5)
                except (OSError, RuntimeError) as e:
                    logger.exception(f"Error while joining process {idx} during finalize: {e}")
                except Exception as e:
                    logger.exception(f"Unexpected error while joining process {idx} during finalize: {e}")

        self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Cleans up all process and pipe resources.

        This private method terminates all running processes and closes all pipes.
        It tracks any failures during cleanup and raises a RuntimeError if cleanup
        fails for any of the resources.

        Raises:
            RuntimeError: If any process or pipe fails to be cleaned up properly.
        """
        failure_processes = []
        for idx, process in enumerate(self.process_pool):
            if process.is_alive():
                logger.warning(f"Terminating process {idx}...")
                try:
                    process.terminate()
                    process.join(timeout=2)
                except (OSError, RuntimeError) as e:
                    logger.exception(f"Failed to terminate process {idx}: {e}")
                    failure_processes.append(idx)
                except Exception as e:
                    logger.exception(f"Unexpected error while terminating process {idx} during cleanup resources: {e}")
                    failure_processes.append(idx)

        failure_pipes = []
        for idx, pipe in enumerate(self.parent_pipes):
            try:
                pipe.close()
            except (BrokenPipeError, EOFError, OSError) as e:
                logger.exception(f"Failed to close {idx} parent pipe cleanly: {e}")
                failure_pipes.append(idx)
            except Exception as e:
                logger.exception(f"Unexpected error while closing parent pipe {idx}: {e}")
                failure_pipes.append(idx)

        self.process_pool = []
        self.parent_pipes = []

        if failure_processes or failure_pipes:
            raise RuntimeError(f"Failed to cleanup subprocesses: {failure_processes}, parent pipes: {failure_pipes}!")

    def is_ready(self):
        """Checks if all inference processes are ready and healthy.

        This method verifies that all processes are alive and responsive by sending
        a CHECK message to each and waiting for a response. If any process is not
        responsive or returns an error, it raises a RuntimeError and cleans up resources.

        Raises:
            RuntimeError: If model is not loaded, if any process is not alive,
                        or if health check fails on any device.
        """
        if not self.process_pool:
            raise RuntimeError("Model not loaded, please check!")

        for idx, process in enumerate(self.process_pool):
            if not process.is_alive():
                self._cleanup_resources()
                raise RuntimeError(f"Process {idx} is not alive, please check")

        for idx, pipe in enumerate(self.parent_pipes):
            try:
                pipe.send((ProcessMessage.CHECK, ""))
            except (BrokenPipeError, EOFError, OSError) as e:
                self._cleanup_resources()
                raise RuntimeError(f"Failed to send CHECK to device {idx}: {e}") from e
            except Exception as e:
                self._cleanup_resources()
                raise RuntimeError(f"Unexpected error when sending CHECK to device {idx}: {e}") from e

        for idx, pipe in enumerate(self.parent_pipes):
            try:
                msg_type, msg = pipe.recv()
                if msg_type == ProcessMessage.ERROR:
                    self._cleanup_resources()
                    raise RuntimeError(f"Device {idx} health check failed: {msg}, cleanup all resource!")
            except EOFError as e:
                self._cleanup_resources()
                raise RuntimeError(f"Device {idx} connection closed unexpectedly during health check.") from e


class InferenceModel:
    """A class for performing inference using MindSpore Lite.

    This class initializes a MindSpore Lite model on a specific device and provides
    methods for running inference, dummy inference, and accessing model properties.

    Attributes:
        model: MindSpore Lite Model instance.
        input_shape_list (list): List of input shapes for the model.
        model_batch_size (int): Batch size of the model, extracted from the first input shape.
    """

    def __init__(self, model_path: str, device_id: int):
        """Initializes the InferenceModel instance.

        Creates a MindSpore Lite context for the specified device, loads the model
        from the given path, and extracts input shapes and batch size information.

        Args:
            model_path (str): Path to the MindIR model file.
            device_id (int): ID of the device to run inference on (e.g. Ascend NPU ID).

        Raises:
            RuntimeError: If the loaded model has no input.
        """
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = device_id

        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
        self.input_shape_list = [item.shape for item in self.model.get_inputs()]
        if not self.input_shape_list:
            raise RuntimeError("The loaded model has no input!")
        self.model_batch_size = self.input_shape_list[0][0]

    def infer(self, batch_inputs: List[np.ndarray]):
        """Performs inference on the provided batch of input data.

        This method takes a list of numpy arrays as input, assigns them to the model's
        input tensors, runs the prediction, and returns the output as a list of numpy arrays.

        Args:
            batch_inputs (List[np.ndarray]): List of input data as numpy arrays. The number
                                           of inputs should match the number of model inputs.

        Returns:
            list: List of inference results as numpy arrays.

        Raises:
            ValueError: If the number of inputs doesn't match the number of model inputs.
        """
        inputs = self.model.get_inputs()
        if len(batch_inputs) != len(inputs):
            raise ValueError(f"The number of model inputs {len(inputs)} should be the same as \
                             the number of inputs {len(batch_inputs)}")

        for i, input_ in enumerate(inputs):
            input_.set_data_from_numpy(batch_inputs[i])

        batch_out = self.model.predict(inputs)
        return [item.get_data_to_numpy() for item in batch_out]

    def dummy_infer(self):
        """Performs a dummy inference without input data.

        This method runs the model prediction without providing specific input data,
        which is typically used for model warm-up or health checks.

        Returns:
            list: Model outputs from the prediction.
        """
        inputs = self.model.get_inputs()
        outputs = self.model.predict(inputs)
        return outputs

    @property
    def batch_size(self):
        """int: The batch size of the model, extracted from the first input shape."""
        return self.model_batch_size

    @property
    def input_shapes(self):
        """list: List of input shapes for the model."""
        return self.input_shape_list
