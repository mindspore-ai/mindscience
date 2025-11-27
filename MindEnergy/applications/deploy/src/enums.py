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
Enumeration module for MindScience deployment service.

This module defines various enumerations used throughout the
MindScience deployment system, including:

- ProcessMessage: Message types for inter-process communication
- HealthStatus: Health status indicators for services
- ModelStatus: Status indicators for model loading and execution
- TaskStatus: Execution status indicators for inference tasks

These enums provide type-safe constants for status values and message types
used in the system's processes and communication.
"""

from enum import Enum, IntEnum

class ProcessMessage(IntEnum):
    """Enumeration for process message types in the system."""
    ERROR = 0
    PREDICT = 1
    REPLY = 2
    CHECK = 3
    BUILD_FINISH = 4
    EXIT = 5


class HealthStatus(str, Enum):
    """Enumeration for health status of services."""
    READY = "ready"
    NOTLOADED = "not_loaded"
    EXCEPTION = "exception"


class ModelStatus(str, Enum):
    """Enumeration for model loading or execution status."""
    SUCCESS = "success"
    FAILURE = "failure"


class TaskStatus(str, Enum):
    """Enumeration for task execution status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
