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
Schemas module for MindScience deployment service.

This module defines Pydantic data models (schemas) used for request/response
validation and data serialization in the MindScience deployment service.
Currently includes:

- ModelInfo: A schema for representing model status and associated messages
  returned by the deployment service APIs

These schemas ensure type safety and proper validation of data exchanged
between clients and the deployment service.
"""

from pydantic import BaseModel

from .enums import ModelStatus

class ModelInfo(BaseModel):
    """Model information containing status and message.

    Attributes:
        status: The status of the model, defaults to ModelStatus.SUCCESS.
        message: The message associated with the model status, defaults to empty string.
    """
    status: str = ModelStatus.SUCCESS
    message: str = ""
