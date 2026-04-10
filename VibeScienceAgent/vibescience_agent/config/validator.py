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
"""Configuration validation utilities for VibeScienceAgent config system."""
from typing import Any, Optional, Union, Type


def check_not_none(config_name: str, param_name: str, value: Any):
    """Check that a value is not None."""
    if value is None:
        raise ValueError(f"{param_name} for {config_name} cannot be None")


def check_number_range(
    param_name: str,
    value: Any,
    min_value: Optional[Any] = None,
    max_value: Optional[Any] = None,
    include_min: bool = True,
    include_max: bool = True,
):
    """Check if a numeric value is within a specified range."""
    if min_value is not None:
        if include_min and value < min_value:
            raise ValueError(f"{param_name} {value} is below minimum allowed value {min_value}")
        if not include_min and value <= min_value:
            raise ValueError(f"{param_name} {value} must be greater than minimum value {min_value}")

    if max_value is not None:
        if include_max and value > max_value:
            raise ValueError(f"{param_name} {value} is above maximum allowed value {max_value}")
        if not include_max and value >= max_value:
            raise ValueError(f"{param_name} {value} must be less than maximum value {max_value}")


def check_type(param_name: str, value: Any, expected_type: Union[Type, tuple[Type, ...]], allow_subclass: bool = True):
    """Check if a value is of the expected type."""
    if isinstance(expected_type, tuple):
        types = expected_type
    else:
        types = (expected_type,)

    if allow_subclass:
        if not isinstance(value, types):
            type_names = ", ".join(t.__name__ for t in types)
            raise ValueError(f"{param_name} {value!r} is not an instance of ({type_names})")
    else:
        type_names = ", ".join(t.__name__ for t in types)
        actual_type = type(value).__name__
        if not isinstance(value, tuple(types)):
            raise ValueError(f"{param_name} {value!r} has type '{actual_type}', expected {type_names}")


def check_is_positive(param_name: str, value: Any):
    """Check if a numeric value is positive."""
    if value <= 0:
        raise ValueError(f"{param_name} {value} must be positive.")


def check_non_negative(param_name: str, value: Any):
    """Check if a numeric value is non-negative."""
    if value < 0:
        raise ValueError(f"{param_name} {value} must be non-negative.")


def check_list_subset(param_name: str, list1: list, list2: list):
    """Check if list1 is a subset of list2 (all elements in list1 exist in list2)."""
    set2 = set(list2)
    extra_elements = [item for item in list1 if item not in set2]

    if extra_elements:
        raise ValueError(
            f"{param_name} contains elements not in allowed list: {extra_elements}. "
            f"Allowed elements: {list2}"
        )
