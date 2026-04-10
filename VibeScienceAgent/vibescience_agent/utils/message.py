# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Custom message types for VibeScienceAgent."""
from typing import TypedDict


class Message(TypedDict):
    """Simple dict-based message format for storing messages in context."""
    role: str  # "user" or "assistant"
    content: str

def create_user_msg(content: str) -> Message:
    """Create a user message."""
    return Message(role="user", content=content)

def create_assistant_msg(content: str) -> Message:
    """Create an assistant message."""
    return Message(role="assistant", content=content)
