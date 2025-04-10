"""
Environment configuration and detection utilities for AI Platform Trainer.

This module provides functions to detect and configure the runtime environment,
including GPU availability, CUDA versions, and fallback mechanisms.
"""
import logging
import platform
import subprocess
import sys
from typing import Dict, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class EnvironmentInfo:
    """Holds information about the runtime environment."""

    def __init__(self) -> None:
        """Initialize environment info."""
        self.system: str = platform.system()
        self.python_version: str = platform.python_version()
        self.cuda_available: bool = torch.cuda.is_available()
        self.cuda_version: Optional[str] = None
        self.gpu_name: Optional[str] = None
        self.gpu_count: int = 0
        self.device_capabilities: Dict[str, Union[str, int, float]] = {}
        self.nvcc_available: bool = False
        self.is_initialized: bool = False

    def detect(self) -> "EnvironmentInfo":
        """Detect and populate environment information."""
        # Detect CUDA capabilities
        if self.cuda_available:
            self.cuda_version = torch.version.cuda
            self.gpu_count = torch.cuda.device_count()
            if self.gpu_count > 0:
                self.gpu_name = torch.cuda.get_device_name(0)
                # Get device capabilities
                device_properties = torch.cuda.get_device_properties(0)
                self.device_capabilities = {
                    "compute_capability": f"{device_properties.major}.{device_properties.minor}",
                    "total_memory": device_properties.total_memory / (1024**3),  # Convert to GB
                    "multi_processor_count": device_properties.multi_processor_count,
                }

        # Check if nvcc is available (for direct CUDA compilation)
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, check=False
            )
            self.nvcc_available = result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            self.nvcc_available = False

        self.is_initialized = True
        return self

    def to_dict(self) -> Dict[str, Union[str, bool, int, Dict]]:
        """Convert environment info to dictionary."""
        if not self.is_initialized:
            self.detect()

        return {
            "system": self.system,
            "python_version": self.python_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "device_capabilities": self.device_capabilities,
            "nvcc_available": self.nvcc_available,
        }

    def __str__(self) -> str:
        """Return string representation of environment info."""
        if not self.is_initialized:
            self.detect()

        lines = [
            "AI Platform Trainer Environment:",
            f"- System: {self.system}",
            f"- Python: {self.python_version}",
            f"- CUDA available: {self.cuda_available}",
        ]

        if self.cuda_available:
            lines.extend(
                [
                    f"- CUDA version: {self.cuda_version}",
                    f"- GPU count: {self.gpu_count}",
                    f"- GPU name: {self.gpu_name}",
                ]
            )
            if self.device_capabilities:
                lines.append("- Device capabilities:")
                for key, value in self.device_capabilities.items():
                    lines.append(f"  - {key}: {value}")

        lines.append(f"- NVCC available: {self.nvcc_available}")
        return "\n".join(lines)


def get_device() -> torch.device:
    """
    Get the appropriate device for PyTorch operations.

    Returns:
        torch.device: The device to use for PyTorch operations (CUDA if available, CPU otherwise)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_cuda_tensor_allocation() -> Tuple[bool, Optional[str]]:
    """
    Test if tensor allocation on GPU works correctly.

    Returns:
        Tuple[bool, Optional[str]]: Success status and error message if any
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    try:
        x = torch.rand(5, 3).cuda()
        y = x + x  # Perform a simple operation
        y.cpu()  # Move back to CPU
        return True, None
    except Exception as e:
        return False, str(e)


def setup_gpu_environment() -> Tuple[torch.device, bool]:
    """
    Set up the GPU environment and return the appropriate device.

    Returns:
        Tuple[torch.device, bool]: The device to use and whether CUDA is working
    """
    env_info = EnvironmentInfo().detect()
    device = get_device()
    cuda_working = False

    if env_info.cuda_available:
        success, error_msg = test_cuda_tensor_allocation()
        if success:
            logger.info("CUDA is available and working correctly")
            cuda_working = True
        else:
            logger.warning(f"CUDA is available but tensor operations failed: {error_msg}")
            logger.warning("Falling back to CPU")
    else:
        logger.info("CUDA is not available, using CPU")

    return device, cuda_working


def print_environment_info() -> None:
    """Print environment information to stdout."""
    env_info = EnvironmentInfo().detect()
    print(env_info)


def main() -> int:
    """Run environment detection and print information."""
    print_environment_info()
    
    device, cuda_working = setup_gpu_environment()
    print(f"\nSelected device: {device}")
    
    if torch.cuda.is_available():
        success, error = test_cuda_tensor_allocation()
        if success:
            print("✓ GPU tensor operations: Working")
        else:
            print(f"✗ GPU tensor operations: Failed - {error}")
    
    return 0 if cuda_working or not torch.cuda.is_available() else 1


if __name__ == "__main__":
    sys.exit(main())
