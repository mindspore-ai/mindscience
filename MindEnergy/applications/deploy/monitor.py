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
MindScience Monitoring Service.

This module implements a server resource monitoring service based on FastAPI that supports:
- Real-time monitoring of CPU usage
- Real-time monitoring of memory usage
- Real-time monitoring of NPU (Neural Processing Unit) usage
- Real-time monitoring of NPU memory usage
- Health check endpoint for server resource usage

The service provides a RESTful API endpoint to retrieve current system resource statistics,
which can be used for system health monitoring and performance analysis.
"""

import signal
import asyncio
from contextlib import asynccontextmanager

import psutil
from loguru import logger
from fastapi import FastAPI
from uvicorn import Server
from uvicorn.config import Config

from src.utils import Utilities
from src.config import configure_logging, ServerConfig

# pylint: disable=unused-argument, redefined-outer-name

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifespan by initializing and cleaning up resources.

    This function is used as an async context manager for the FastAPI application
    lifespan. It handles logging configuration and performs startup and shutdown
    logging.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    configure_logging("monitor")
    logger.info("Starting monitor service.")
    yield
    logger.info("Shutting down monitor service.")

app = FastAPI(lifespan=lifespan)

@app.get("/mindscience/monitor/resource_usage")
async def get_server_status():
    """Retrieves the current server resource usage statistics.

    This endpoint collects and returns CPU, memory, and NPU usage statistics
    from the server. It uses psutil for CPU and memory metrics and a custom
    utility function for NPU metrics.

    Returns:
        dict: A dictionary containing the status, message, and resource usage data
        including CPU usage rate, memory usage rate, NPU usage rate, and NPU memory usage rate.
    """
    try:
        results = await asyncio.gather(
            asyncio.to_thread(psutil.cpu_percent, interval=None),
            asyncio.to_thread(lambda: psutil.virtual_memory().percent),
            Utilities.get_npu_usage(),
            return_exceptions=True
        )

        # 解包结果
        cpu_usage, memory_usage, npu_stats = results

        if isinstance(cpu_usage, Exception):
            logger.error(f"Failed to get CPU usage: {cpu_usage}")
            cpu_usage = 0.0

        if isinstance(memory_usage, Exception):
            logger.error(f"Failed to get memory usage: {memory_usage}")
            memory_usage = 0.0

        if isinstance(npu_stats, Exception):
            logger.warning(f"Failed to get NPU usage info: {npu_stats}")
            npu_usage_rate = "0.00"
            npu_memory_usage_rate = "0.00"
        elif npu_stats:
            total_memory = sum(stat["memory_total_mb"] for stat in npu_stats)
            used_memory = sum(stat["memory_used_mb"] for stat in npu_stats)
            avg_utilization = sum(stat["utilization_percent"] for stat in npu_stats) / len(npu_stats)
            avg_memory_usage = (used_memory / total_memory * 100) if total_memory > 0 else 0
            npu_usage_rate = f"{avg_utilization:.2f}"
            npu_memory_usage_rate = f"{avg_memory_usage:.2f}"
        else:
            npu_usage_rate = "0.00"
            npu_memory_usage_rate = "0.00"

        return {
            "status": 200,
            "msg": "success",
            "data": {
                "cpuUsageRate": f"{cpu_usage:.2f}",
                "memoryUsageRate": f"{memory_usage:.2f}",
                "npuUsageRate": npu_usage_rate,
                "npuMemoryUsageRate": npu_memory_usage_rate
            }
        }
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_server_status: {e}")
        return {
            "status": 500,
            "msg": "failure",
            "data": {}
        }

if __name__ == "__main__":
    server = Server(
        Config(
            app=app,
            host=ServerConfig.host,
            port=ServerConfig.monitor_port,
            limit_concurrency=ServerConfig.limit_concurrency,
            timeout_keep_alive=ServerConfig.timeout_keep_alive,
            backlog=ServerConfig.backlog
        )
    )

    def terminate_signal_handler(signum, frame):
        """Handles termination signals to gracefully shut down the server.

        This function is called when the server receives a termination signal
        (SIGTERM or SIGINT). It sets the server to exit.

        Args:
            signum (int): The signal number received.
            frame: The current stack frame (unused).
        """
        logger.info(f"Catch signal: {signum}, starting terminate server...")
        server.should_exit = True

    signal.signal(signal.SIGTERM, terminate_signal_handler)
    signal.signal(signal.SIGINT, terminate_signal_handler)


    logger.info("Starting monitor server...")
    server.run()
