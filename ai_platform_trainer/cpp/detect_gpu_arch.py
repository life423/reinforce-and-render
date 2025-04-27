#!/usr/bin/env python
"""
GPU Architecture detection script.

This script detects the available GPU architecture(s) and determines 
compatible CUDA architectures for compilation. It handles cases where
the installed PyTorch version doesn't directly support the current GPU's
architecture by finding the closest supported version.
"""
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_torch_supported_architectures():
    """
    Get CUDA architectures supported by the installed PyTorch version.
    
    Returns:
        List of supported architecture codes (e.g., ["70", "75", "80"])
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available in PyTorch.")
            return []
        
        # Get PyTorch's CUDA capabilities string from error message
        # This is a bit of a hack but works reliably
        try:
            # Try to get capabilities by intentionally causing a helpful error
            torch.zeros(1, device="cuda:0", dtype=torch.half)
        except RuntimeError as e:
            error_msg = str(e)
            if "GPU capability" in error_msg:
                # Extract capabilities from the error message
                # Example: "sm_37 sm_50 sm_60" etc.
                for line in error_msg.split("\n"):
                    if "sm_" in line:
                        # Extract numerical parts of sm_XX
                        arch_list = [
                            s.replace("sm_", "") 
                            for s in line.split() 
                            if s.startswith("sm_")
                        ]
                        logging.info(f"PyTorch CUDA architectures: {', '.join(arch_list)}")
                        return arch_list
        
        # Fallback method if error message approach fails
        # Common architectures supported by recent PyTorch versions
        logging.warning("Couldn't determine PyTorch CUDA architectures, using defaults.")
        return ["70", "75", "80", "86", "90"]
    except ImportError:
        logging.error("PyTorch not installed, cannot determine CUDA architectures.")
        return []


def get_current_gpu_arch():
    """
    Get the architecture of the current CUDA GPU.
    
    Returns:
        String with architecture code (e.g., "80") or None if unavailable
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logging.warning("No CUDA GPU available.")
            return None
        
        # Get compute capability
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"{major}{minor}"
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"Detected GPU: {device_name} with architecture sm_{arch}")
        
        return arch
    except ImportError:
        logging.error("PyTorch not installed, cannot detect GPU architecture.")
        return None
    except Exception as e:
        logging.error(f"Error detecting GPU architecture: {e}")
        return None


def find_nearest_supported_arch(current_arch, supported_archs):
    """
    Find the nearest supported architecture that's <= the current arch.
    
    Args:
        current_arch: Current GPU architecture code (e.g., "120")
        supported_archs: List of supported architecture codes
    
    Returns:
        Best supported architecture code
    """
    if not current_arch or not supported_archs:
        return None
        
    current_num = int(current_arch)
    
    # Convert supported archs to integers for comparison
    supported_nums = [int(a) for a in supported_archs]
    
    # Find the nearest architecture that's less than or equal to current
    # Since older architectures, we want the highest that's <= current
    valid_archs = [a for a in supported_nums if a <= current_num]
    
    if valid_archs:
        nearest = max(valid_archs)
        logging.info(
            f"Selected nearest supported architecture: sm_{nearest} "
            f"(current: sm_{current_arch})"
        )
        return str(nearest)
    else:
        # If no supported arch is <= current, fall back to the lowest supported one
        nearest = min(supported_nums)
        logging.warning(
            f"Current architecture sm_{current_arch} not supported. "
            f"Falling back to sm_{nearest}."
        )
        return str(nearest)


def generate_cmake_arch_flags():
    """
    Generate appropriate architecture flags for CMake.
    
    This will include both the current architecture (if supported by PyTorch)
    and a set of common architectures for best compatibility.
    
    Returns:
        String in CMake list format: "70;75;80;86"
    """
    # Get all relevant architectures
    supported_archs = get_torch_supported_architectures()
    current_arch = get_current_gpu_arch()
    
    # Start with a base set of common architectures
    base_archs = ["70", "75", "80", "86"]
    
    # Add current architecture if available and supported
    if current_arch:
        best_arch = find_nearest_supported_arch(current_arch, supported_archs)
        if best_arch and best_arch not in base_archs:
            base_archs.append(best_arch)
    
    # Return as semicolon-separated list for CMake
    return ';'.join(base_archs)


def main():
    """Main entry point."""
    try:
        # Generate flag for CMake
        cmake_flags = generate_cmake_arch_flags()
        
        # Print to stdout for CMake to capture
        print(cmake_flags)
        
        return 0
    except Exception as e:
        logging.error(f"Error generating architecture flags: {e}")
        # Return a default set if we encounter any errors
        print("70;75;80")
        return 1


if __name__ == "__main__":
    sys.exit(main())
