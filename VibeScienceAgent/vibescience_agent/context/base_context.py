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
"""Base context class for VibeScienceAgent context management."""
import abc


class BaseContext(abc.ABC):
    """Abstract base class for context management in VibeScienceAgent."""
    def __init__(self):
        """Initialize base context."""
        self.context = {}

    def _asdict(self):
        """Convert context to dictionary format."""
        return {"context": self.context}

    def add_context(self, name, value):
        """Add context value by name."""
        pass

    def get_context(self, name):
        """Get context value by name."""
        pass
