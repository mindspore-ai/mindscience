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
Session management module for MindScience deployment service.

This module provides the SessionManager class for managing model inference sessions.
It includes functionality for:

- Loading and initializing models for inference
- Managing multiple inference sessions across different devices
- Performing batch inference operations
- Checking model health status
- Unloading models and cleaning up resources

The SessionManager handles the lifecycle of model sessions, from initialization
to finalization, ensuring proper resource management and inference execution.
"""

import numpy as np
from loguru import logger

from .utils import Utilities
from .enums import HealthStatus
from .config import DeployConfig
from .inference import MultiprocessInference

class SessionManager:
    """Manages model sessions for inference tasks.

    This class handles the initialization, deletion, and inference operations
    for machine learning models, including loading models, managing sessions,
    and performing batch inference.
    """

    def __init__(self):
        """Initializes the SessionManager with default values."""
        self.current_model = ""
        self.device_num = -1
        self.infer_sessions = None
        self._model_dict = {}

    def init_session(self, model_name, device_num):
        """Initializes a model session for inference.

        Collects model files, validates device number and model name,
        creates inference sessions, and builds the model for each session.

        Args:
            model_name (str): Name of the model to load.
            device_num (int): Number of devices to use for inference.

        Raises:
            ValueError: If device number exceeds maximum, model is not found,
                       or model is already loaded.
            RuntimeError: If session initialization fails.
        """
        self._model_dict = Utilities.collect_mindir_models(DeployConfig.models_dir)

        if device_num > DeployConfig.max_device_num:
            raise ValueError(f"Device num {device_num} is over the actual device num {DeployConfig.max_device_num}, \
                             please check!")
        if model_name not in self.model_dict:
            raise ValueError(f"model {model_name} is not in {self.model_dict.keys()}, please check!")
        if model_name == self.current_model:
            raise ValueError(f"model {model_name} is already loaded, please check!")

        model_paths = self.model_dict.get(model_name)
        sessions = [MultiprocessInference(model_path, device_num) for model_path in model_paths]
        try:
            for i, session in enumerate(sessions):
                session.build_model()
                logger.info(f"{model_paths[i]} is loaded successfully, {device_num} sessions are inited")
        except Exception as e:
            raise RuntimeError(f"Init session failed, please check! ERROR: {e}") from e

        self.current_model = model_name
        self.device_num = device_num
        self.infer_sessions = sessions

    def del_session(self):
        """Deletes the current model session.

        Cleans up the current model, device number, and model dictionary.
        Finalizes all inference sessions if they exist.

        Raises:
            RuntimeError: If deleting the session fails.
        """
        self.current_model = ""
        self.device_num = -1
        self._model_dict = {}

        if self.infer_sessions is None:
            logger.warning("Session not inited, please check!")
        else:
            try:
                for session in self.infer_sessions:
                    session.finalize()
                self.infer_sessions = None
                logger.info("Session is deleted successfully")
            except Exception as e:
                raise RuntimeError(f"Delete session failed, please check! ERROR: {e}") from e

    def batch_infer(self, batch_inputs, task_type):
        """Performs batch inference on the given inputs.

        Processes batch inputs by extending them to match model batch size,
        and performs inference using the specified task type session.

        Args:
            batch_inputs (list): List of input arrays for batch inference.
            task_type (int): Index of the inference session to use.

        Returns:
            list: List of concatenated results from the inference.

        Raises:
            ValueError: If inputs are empty, task type is invalid, or
                       model has not been loaded.
            RuntimeError: If model has not been loaded.
        """
        if not batch_inputs:
            raise ValueError("Input is empty, please check!")
        if self.infer_sessions is None:
            raise RuntimeError("Model has not been loaded, please check!")
        if task_type >= len(self.infer_sessions):
            raise ValueError(f"Only task_type {list(range(len(self.infer_sessions)))} is supported, \
                             but given {task_type}")

        session = self.infer_sessions[task_type]
        data_size = batch_inputs[0].shape[0]
        model_batch_size = session.batch_size

        batch_inputs = [Utilities.extend_input(item, model_batch_size) for item in batch_inputs]
        total_size = batch_inputs[0].shape[0]

        input_datas = []
        result_list = []
        for i in range(0, total_size, model_batch_size):
            input_datas.append([item[i: i + model_batch_size] for item in batch_inputs])
            if len(input_datas) > 0 and len(input_datas) % self.device_num == 0:
                result_list = self._infer(session, input_datas, result_list)
                input_datas = []
        if input_datas:
            result_list = self._infer(session, input_datas, result_list)

        model_out_num = len(result_list[0])
        final_ret = []
        for i in range(model_out_num):
            final_ret.append(np.concatenate([result[i] for result in result_list], axis=0)[:data_size])
        return final_ret

    def _infer(self, session, input_datas, result_list):
        """Performs inference on input data using the given session.

        Internal helper method that executes the actual inference and
        extends the result list with predicted results.

        Args:
            session (MultiprocessInference): Inference session to use.
            input_datas (list): List of input data to be inferred.
            result_list (list): List to extend with inference results.

        Returns:
            list: Updated result list with new inference results.

        Raises:
            RuntimeError: If the model prediction fails.
        """
        try:
            predict_result = session.infer(input_datas)
            result_list.extend(predict_result)
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(f"{self.current_model} predict failed, please check! ERROR: {e}") from e
        return result_list

    def is_ready(self):
        """Checks if the model sessions are ready for inference.

        Verifies if inference sessions are initialized and operational.

        Returns:
            HealthStatus: Status indicating if model is NOTLOADED, READY,
                          or in EXCEPTION state.
        """
        if self.infer_sessions is None:
            logger.info("Model has not been loaded, please check!")
            return HealthStatus.NOTLOADED
        try:
            for session in self.infer_sessions:
                session.is_ready()
            logger.info(f"Model: {self.current_model} is ready!")
            return HealthStatus.READY
        except Exception:
            logger.info(f"Model: {self.current_model} is unavailable!")
            return HealthStatus.EXCEPTION

    @property
    def session_name(self):
        """str: Name of the currently loaded model."""
        return self.current_model

    @property
    def model_dict(self):
        """dict: Dictionary containing model information."""
        return self._model_dict
