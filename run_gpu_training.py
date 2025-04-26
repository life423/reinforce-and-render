#!/usr/bin/env python
"""
GPU training wrapper script.

This script performs necessary checks, builds CUDA extensions, and runs
the enemy RL model training with GPU acceleration.
"""
import os
import sys
import subprocess
import argparse
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gpu_training_setup.log')
    ]
)

def check_cuda_environment():
    """
    Check if CUDA is available in the current environment.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logging.info(f"[SUCCESS] CUDA is available: {gpu_name}, CUDA version: {cuda_version}")
            return True
        else:
            logging.warning("[WARNING] CUDA is NOT available! Training will use CPU only (slower).")
            return False
    except ImportError:
        logging.error("[ERROR] PyTorch not installed. Please install it with CUDA support.")
        return False

def check_nvcc_availability():
    """
    Check if the NVIDIA CUDA compiler (nvcc) is available.
    
    Returns:
        bool: True if nvcc is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            nvcc_version = result.stdout.strip()
            logging.info(f"[SUCCESS] NVIDIA CUDA Compiler (nvcc) is available:\n{nvcc_version}")
            return True
        else:
            logging.warning("[WARNING] NVIDIA CUDA Compiler (nvcc) check failed")
            return False
    except Exception:
        logging.warning("[WARNING] NVIDIA CUDA Compiler (nvcc) not found in PATH")
        return False

def build_cuda_extensions():
    """
    Build the CUDA C++ extensions.
    
    Returns:
        bool: True if build was successful, False otherwise
    """
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir  # Assuming this script is in the project root
    cpp_dir = os.path.join(root_dir, 'ai_platform_trainer', 'cpp')
    
    if not os.path.exists(cpp_dir):
        logging.error(f"[ERROR] C++ directory not found: {cpp_dir}")
        return False
    
    logging.info(f"Building CUDA extensions in {cpp_dir}...")
    
    # Windows: Use build.bat
    if os.name == 'nt' and os.path.exists(os.path.join(cpp_dir, 'build.bat')):
        os.chdir(cpp_dir)
        try:
            result = subprocess.run(
                ['build.bat'], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                logging.error(f"[ERROR] build.bat failed:\n{result.stderr}")
                return False
                
            logging.info(f"[SUCCESS] Successfully built CUDA extensions with build.bat")
            os.chdir(root_dir)  # Return to original directory
            return True
        except Exception as e:
            logging.error(f"[ERROR] Error running build.bat: {e}")
            os.chdir(root_dir)  # Return to original directory
            return False
    # All platforms: Use setup.py
    else:
        try:
            os.chdir(cpp_dir)
            result = subprocess.run(
                [sys.executable, 'setup.py', 'build_ext', '--inplace'], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                logging.error(f"[ERROR] CUDA extension build failed:\n{result.stderr}")
                return False
                
            logging.info(f"[SUCCESS] Successfully built CUDA extensions with setup.py")
            os.chdir(root_dir)  # Return to original directory
            return True
        except Exception as e:
            logging.error(f"[ERROR] Error building CUDA extensions: {e}")
            os.chdir(root_dir)  # Return to original directory
            return False

def verify_cuda_build():
    """
    Verify that the CUDA extension was built and is importable.
    
    Returns:
        bool: True if verification passed, False otherwise
    """
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from ai_platform_trainer.cpp.gpu_environment import HAS_GPU_ENV
        
        if HAS_GPU_ENV:
            logging.info("[SUCCESS] GPU environment extension successfully imported")
            return True
        else:
            logging.warning("[WARNING] GPU environment module found but not properly built with CUDA")
            return False
    except ImportError as e:
        logging.error(f"[ERROR] Failed to import GPU environment: {e}")
        logging.error("The CUDA extension may not have been built correctly.")
        return False

def run_training(args):
    """
    Run the enemy RL model training with GPU acceleration.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Return code (0 for success, non-zero for failure)
    """
    # Construct command
    cmd = [
        sys.executable,
        'train_enemy_rl_model.py',
        f'--timesteps={args.timesteps}',
        f'--save-path={args.save_path}',
        f'--log-path={args.log_path}'
    ]
    
    if args.headless:
        cmd.append('--headless')
        
    if args.force_cpu:
        cmd.append('--force-cpu')
        
    if args.verify_gpu:
        cmd.append('--verify-gpu')
    
    # Run the command
    logging.info(f"Running training command: {' '.join(cmd)}")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream the output
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    return_code = process.wait()
    
    # Calculate duration
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if return_code == 0:
        logging.info(f"[SUCCESS] Training completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    else:
        logging.error(f"[ERROR] Training failed with return code {return_code}")
    
    return return_code

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train enemy AI using GPU acceleration')
    parser.add_argument(
        '--timesteps',
        type=int,
        default=500000,
        help='Number of timesteps to train for (default: 500000)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run training without visualization (faster training)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='models/enemy_rl',
        help='Directory to save the model to (default: models/enemy_rl)'
    )
    parser.add_argument(
        '--log-path',
        type=str,
        default='logs/enemy_rl',
        help='Directory to save TensorBoard logs to (default: logs/enemy_rl)'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force using CPU even if GPU is available'
    )
    parser.add_argument(
        '--verify-gpu',
        action='store_true',
        help='Verify GPU is being effectively used during training'
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip building CUDA extensions (use existing build)'
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    print("=" * 80)
    print("  GPU-Accelerated Enemy Training Pipeline")
    print("=" * 80)
    
    args = parse_args()
    
    # Step 1: Check CUDA environment
    cuda_available = check_cuda_environment()
    if not cuda_available and not args.force_cpu:
        user_input = input("CUDA is not available. Continue with CPU training? (y/n): ")
        if user_input.lower() != 'y':
            logging.info("Training aborted by user.")
            return 1
    
    # Step 2: Check NVCC
    if cuda_available and not args.force_cpu and not args.skip_build:
        nvcc_available = check_nvcc_availability()
        if not nvcc_available:
            logging.warning("NVCC not found, which may affect CUDA extension building.")
    
    # Step 3: Build CUDA extensions
    if not args.skip_build:
        if cuda_available and not args.force_cpu:
            build_success = build_cuda_extensions()
            if not build_success:
                logging.warning("CUDA extension build failed. Will attempt to continue with existing build.")
        else:
            logging.info("Skipping CUDA extension build as CUDA is not available or CPU was forced.")
    else:
        logging.info("Skipping CUDA extension build as requested.")
    
    # Step 4: Verify the build
    if cuda_available and not args.force_cpu:
        verify_success = verify_cuda_build()
        if not verify_success:
            logging.warning("CUDA extension verification failed. Training may run slower on CPU.")
    
    # Step 5: Run the training
    return run_training(args)

if __name__ == "__main__":
    sys.exit(main())
