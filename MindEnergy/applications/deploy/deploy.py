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
MindScience Deployment Service.

This module implements a model deployment service based on FastAPI that supports:
- Model loading/unloading via HTTP interface
- Asynchronous inference execution with background tasks
- Task status management (pending, processing, completed, error)
- Health checking functionality
- Multi-device parallel inference support (up to 8 NPU devices)
- Result file download after inference completion

The service provides RESTful APIs for model management and inference execution,
with proper error handling and resource management.
"""

import os
import uuid
import signal
from typing import Any, Union
from contextlib import asynccontextmanager

from loguru import logger
from uvicorn import Server
from uvicorn.config import Config
from fastapi import FastAPI, Form, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.enums import HealthStatus, ModelStatus, TaskStatus
from src.schemas import ModelInfo
from src.session import SessionManager
from src.utils import Utilities
from src.config import configure_logging, DeployConfig, ServerConfig

# pylint: disable=unused-argument, redefined-outer-name

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifespan by initializing and cleaning up resources.

    This function is used as an async context manager for the FastAPI application
    lifespan. It handles initialization of models on the specified devices during
    startup and performs cleanup during shutdown.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    configure_logging("deploy")
    logger.info(f"Initializing models on devices {list(range(DeployConfig.deploy_device_num))}.")
    yield
    logger.info("Shutting down deploy service.")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"]
)

model = SessionManager()
tasks_status = {}


def inference(dataset_path, task_id, task_type):
    """Performs inference on a dataset.

    This function loads an HDF5 dataset, performs batch inference using the model,
    saves the results to an HDF5 file, and updates the task status.

    Args:
        dataset_path (str): Path to the HDF5 dataset file.
        task_id (str): Unique identifier for the inference task.
        task_type (int): Type of inference task to perform.

    Raises:
        Exception: If there is an error during the inference process.
    """
    try:
        tasks_status[task_id] = TaskStatus.PROCESSING

        batch_inputs = Utilities.load_h5_file(dataset_path)

        results_list = model.batch_infer(batch_inputs, task_type)

        os.makedirs(DeployConfig.results_dir, exist_ok=True)
        results_path = os.path.join(DeployConfig.results_dir, f"{task_id}_results.h5")

        Utilities.save_h5_file(results_list, results_path)

        tasks_status[task_id] = TaskStatus.COMPLETED
    except Exception as e:
        tasks_status[task_id] = TaskStatus.ERROR
        logger.error(f"Task {task_id} infer failed, ERROR: {e}.")


@app.post("/mindscience/deploy/load_model")
async def load_model(model_name: str = Form(), model_file: Union[UploadFile, str] = None):
    """Loads a model for inference.

    This endpoint handles the loading of a model, either from an uploaded file
    or from a local file. If a model is already loaded, it returns an error.

    Args:
        model_name (str): Name of the model to load.
        model_file (Union[UploadFile, str], optional): Uploaded model file. Defaults to None.

    Returns:
        JSONResponse: A response containing model information and status.
    """
    model_info = ModelInfo()
    if model.is_ready() == HealthStatus.READY:
        model_info.status = ModelStatus.FAILURE
        model_info.message = f"Model {model.session_name} is already loaded, please unload first."
        return JSONResponse(
            content=model_info.model_dump(),
            status_code=403,
            headers={"Content-Type": "application/json"}
        )

    if not model_file:
        logger.info("File not uploaded, model loaded from local file.")
    else:
        try:
            logger.info(f"Start receive {model_name} file")
            os.makedirs(DeployConfig.models_dir, exist_ok=True)
            save_dir = os.path.join(DeployConfig.models_dir, model_name)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, model_file.filename)
            await Utilities.save_upload_file(model_file, save_path)

            Utilities.extract_file(save_path)
            logger.info(f"{model_name} file receive successfully.")
        except Exception as e:
            model_info.status = ModelStatus.FAILURE
            model_info.message = str(e)
            return JSONResponse(
                content=model_info.model_dump(),
                status_code=500,
                headers={"Content-Type": "application/json"}
            )

    try:
        model.init_session(model_name, device_num=DeployConfig.deploy_device_num)
        return JSONResponse(
            content=model_info.model_dump(),
            status_code=201,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        model_info.status = ModelStatus.FAILURE
        model_info.message = str(e)
        return JSONResponse(
            content=model_info.model_dump(),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )


@app.post("/mindscience/deploy/unload_model")
async def unload_model():
    """Unloads the currently loaded model.

    This endpoint unloads the model session and frees up the associated resources.

    Returns:
        JSONResponse: A response containing model information and status.
    """
    model_info = ModelInfo()
    try:
        model.del_session()
        return JSONResponse(
            content=model_info.model_dump(),
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        model_info.status = ModelStatus.FAILURE
        model_info.message = str(e)
        return JSONResponse(
            content=model_info.model_dump(),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )


@app.post("/mindscience/deploy/infer")
async def infer(dataset: UploadFile = File(..., description="input dataset"), task_type: int = Form(),
                background_tasks: BackgroundTasks = Any):
    """Performs inference on the uploaded dataset.

    This endpoint accepts a dataset file and performs inference in the background.
    It checks if the model is ready, validates the number of pending tasks,
    saves the uploaded dataset, and starts the inference task.

    Args:
        dataset (UploadFile): The input dataset file to process.
        task_type (int): Type of inference task to perform.
        background_tasks (BackgroundTasks): FastAPI background tasks manager.

    Returns:
        JSONResponse: A response containing the task ID.

    Raises:
        HTTPException: If the server is not ready or if the request limit is exceeded.
    """
    if model.is_ready() != HealthStatus.READY:
        raise HTTPException(status_code=500, detail="Server is not ready, please check")

    pending_number = Utilities.count_pending_task(tasks_status)
    logger.info(f"Pending task number is {pending_number}")
    if pending_number >= DeployConfig.max_request_num:
        logger.error(f"Predict request number exceed limited number: \
                     {pending_number} vs {DeployConfig.max_request_num}")
        raise HTTPException(status_code=503, detail="Request number exceed limited number, please wait for a while.")

    task_id = str(uuid.uuid4())
    tasks_status[task_id] = TaskStatus.PENDING

    try:
        os.makedirs(DeployConfig.datasets_dir, exist_ok=True)
        dataset_path = os.path.join(DeployConfig.datasets_dir, f"{task_id}_{dataset.filename}")
        await Utilities.save_upload_file(dataset, dataset_path)

        background_tasks.add_task(inference, dataset_path, task_id, task_type)

        return JSONResponse(
            content={"task_id": task_id},
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        tasks_status[task_id] = TaskStatus.ERROR
        raise HTTPException(status_code=500, detail=f"Task {task_id} infer failed, ERROR: {e}.") from e


@app.get("/mindscience/deploy/query_status/{task_id}")
async def query_status(task_id: str):
    """Queries the status of an inference task.

    This endpoint retrieves the current status of a specific inference task
    by its task ID.

    Args:
        task_id (str): Unique identifier of the task to query.

    Returns:
        JSONResponse: A response containing the task status.

    Raises:
        HTTPException: If the task ID is not found.
    """
    status = tasks_status.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} is not found, please check!")
    return JSONResponse(
        content={"status": status},
        status_code=200,
        headers={"Content-Type": "application/json"}
    )


@app.get("/mindscience/deploy/query_results/{task_id}")
async def query_results(task_id: str):
    """Retrieves the results of a completed inference task.

    This endpoint returns the results file of a completed inference task
    if the task is completed, otherwise returns the current status.

    Args:
        task_id (str): Unique identifier of the task to query.

    Returns:
        FileResponse or JSONResponse: Either the results file or a response containing the task status and message.

    Raises:
        HTTPException: If the task ID is not found.
    """
    status = tasks_status.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} is not found, please check!")
    if status != TaskStatus.COMPLETED:
        return JSONResponse(
            content={"status": status, "message": f"Task {task_id} is not completed."},
            status_code=404,
            headers={"Content-Type": "application/json"}
        )
    result_path = os.path.join(DeployConfig.results_dir, f"{task_id}_results.h5")
    return FileResponse(result_path, filename=f"{task_id}_results.h5")


@app.get("/mindscience/deploy/health_check")
async def health_check():
    """Performs a health check on the deployment service.

    This endpoint checks the readiness status of the model and returns
    an appropriate HTTP response based on the model's health status.

    Returns:
        JSONResponse: A response indicating the health status of the service.
    """
    health_status = model.is_ready()
    logger.info(f"health check result is {health_status}.")
    if health_status == HealthStatus.NOTLOADED:
        return JSONResponse(
            content={},
            status_code=501,
            headers={"Content-Type": "application/json"}
        )
    if health_status == HealthStatus.EXCEPTION:
        return JSONResponse(
            content={model.session_name: health_status},
            status_code=503,
            headers={"Content-Type": "application/json"}
        )
    return JSONResponse(
        content={model.session_name: health_status},
        status_code=200,
        headers={"Content-Type": "application/json"}
    )


if __name__ == "__main__":
    server = Server(
        Config(
            app=app,
            host=ServerConfig.host,
            port=ServerConfig.deploy_port,
            limit_concurrency=ServerConfig.limit_concurrency,
            timeout_keep_alive=ServerConfig.timeout_keep_alive,
            backlog=ServerConfig.backlog
        )
    )

    def terminate_signal_handler(signum, frame):
        """Handles termination signals to gracefully shut down the server.

        This function is called when the server receives a termination signal
        (SIGTERM or SIGINT). It cleans up the model session and sets the server
        to exit.

        Args:
            signum (int): The signal number received.
            frame: The current stack frame (unused).
        """
        global model
        logger.info(f"Catch signal: {signum}, starting terminate server...")
        try:
            model.del_session()
        except Exception as e:
            logger.exception(f"Model clear failed, please check! ERROR: {e}")
        del model
        server.should_exit = True

    signal.signal(signal.SIGTERM, terminate_signal_handler)
    signal.signal(signal.SIGINT, terminate_signal_handler)

    logger.info("Starting deploy server...")
    server.run()
