"""
Configuration module for CPU/Ascend device management in PowerFlowNet MindSpore version.
This module provides utilities to dynamically switch between CPU and Ascend devices.
"""

import threading
from typing import Literal

import mindspore as ms


class DeviceConfig:
    """Device configuration management with thread-safe singleton pattern."""

    _instance = None
    _lock = threading.Lock()
    _initialized = False
    _device_target: str = 'CPU'
    _device_id: int = 0

    # Supported device types
    SUPPORTED_DEVICES = {'CPU', 'Ascend'}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super(DeviceConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if not DeviceConfig._initialized:
            with DeviceConfig._lock:
                if not DeviceConfig._initialized:
                    DeviceConfig._initialized = True

    @classmethod
    def set_device(cls, device_target: Literal['CPU', 'Ascend'] = 'CPU', device_id: int = 0):
        """
        Set the device for MindSpore computations.

        Args:
            device_target (str): 'CPU' or 'Ascend'. Device type to use for computations.
            device_id (int, optional): Device ID (for multi-device scenarios). Default: 0.

        Examples:
            >>> DeviceConfig.set_device('CPU')
            >>> DeviceConfig.set_device('Ascend', device_id=0)
        """
        if device_target not in cls.SUPPORTED_DEVICES:
            raise ValueError(
                f"Unsupported device: {device_target}. "
                f"Choose from {cls.SUPPORTED_DEVICES}"
            )

        cls._device_target = device_target
        cls._device_id = device_id

        # Set MindSpore context - always use PYNATIVE_MODE for compatibility
        # This is especially important for Ascend devices where GRAPH_MODE
        # can have issues with optimizer compilation
        ms.set_context(mode=ms.PYNATIVE_MODE)

        # For Ascend, also disable JIT compile level optimization to avoid issues
        if device_target == 'Ascend':
            try:
                ms.set_context(jit_config={"jit_level": "O0"})
            except Exception:
                pass  # Older MindSpore versions may not support this

        ms.set_device(device_target=device_target, device_id=device_id)
        print(f"Device set to: {device_target} (ID: {device_id})")

    @classmethod
    def get_device_target(cls) -> str:
        """Get current device target"""
        return cls._device_target

    @classmethod
    def get_device_id(cls) -> int:
        """Get current device ID"""
        return cls._device_id

    @classmethod
    def is_cpu(cls) -> bool:
        """Check if device is CPU"""
        return cls._device_target == 'CPU'

    @classmethod
    def is_ascend(cls) -> bool:
        """Check if device is Ascend"""
        return cls._device_target == 'Ascend'


def init_device(device_target: Literal['CPU', 'Ascend'] = 'CPU', device_id: int = 0):
    """
    Initialize device configuration.

    Args:
        device_target: 'CPU' or 'Ascend'
        device_id: Device ID

    Example:
        >>> init_device('Ascend')
    """
    DeviceConfig.set_device(device_target, device_id)


def get_device_config() -> DeviceConfig:
    """Get the device configuration instance"""
    return DeviceConfig()
