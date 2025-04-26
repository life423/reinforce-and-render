# GPU Training Guide for AI Platform Trainer

This guide explains how to use GPU acceleration for your reinforcement learning model training with the C++ custom modules.

## Prerequisites

Before running the training scripts, ensure you have the following packages installed:

### Option 1: Using an existing conda environment

If you're using the provided `environment-gpu.yml` file:

```bash
# Create and activate the environment
conda env create -f environment-gpu.yml
conda activate ai-platform-gpu

# Additionally, install psutil which is needed for monitoring
conda install psutil
```

### Option 2: Manual installation with pip

If you're using a virtual environment or direct installation:

```bash
# Make sure your virtual environment is activated first, then:
python -m pip install torch numpy matplotlib stable-baselines3 gymnasium psutil

# For GPU support specifically:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

To train your enemy AI model using GPU acceleration, you can use either of these approaches:

### Option 1: Using the GPU Wrapper Script (Recommended)

```bash
python run_gpu_training.py --timesteps 100000 --headless
```

This script will:
1. Check if CUDA is available on your system
2. Build the C++ extensions with CUDA support
3. Verify the GPU environment is properly initialized
4. Run the training with GPU acceleration

### Option 2: Direct Training Script

```bash
python train_enemy_rl_model.py --timesteps 100000 --headless
```

The training script has been modified to automatically detect and use GPU acceleration when available.

## Command Line Arguments

Both scripts support these common arguments:

| Argument | Description |
|----------|-------------|
| `--timesteps` | Number of timesteps to train for (default: 500000) |
| `--headless` | Run without visualization for faster training |
| `--save-path` | Directory to save models to (default: models/enemy_rl) |
| `--log-path` | Directory to save logs to (default: logs/enemy_rl) |
| `--force-cpu` | Force CPU usage even if GPU is available |
| `--verify-gpu` | Verify GPU is being effectively used during training |

Additionally, the wrapper script supports:

| Argument | Description |
|----------|-------------|
| `--skip-build` | Skip building CUDA extensions (use existing build) |

## Verifying GPU Usage

To confirm your training is actually using the GPU:

1. Check the console output during training, which will display:
   ```
   GPU: XX.X%, CPU: YY.Y%, Memory: ZZZ.Z MB
   ```

2. After training completes, examine the resource usage plot at:
   ```
   [log_path]/training_resource_usage.png
   ```

3. If the **maximum GPU utilization** shows below 10%, the GPU may not be properly utilized.

## Troubleshooting

If you experience issues with GPU acceleration:

1. **C++ Extension Build Failure**
   - Ensure you have the CUDA toolkit installed
   - Run `cd ai_platform_trainer/cpp && python setup.py build_ext --inplace`
   - Check for error messages during the build

2. **PyTorch Not Using CUDA**
   - Verify with `python -c "import torch; print(torch.cuda.is_available())"`
   - Reinstall PyTorch with CUDA support: 
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

3. **Missing Python Packages**
   - If you encounter errors about missing packages, install them:
     ```bash
     # Make sure you're in the right environment first, then:
     python -m pip install psutil matplotlib stable-baselines3 gymnasium
     ```
   - If you get "Fatal error in launcher" with pip, use the python module form:
     ```bash
     python -m pip install [package-name]
     ```

3. **Low GPU Utilization**
   - Small models or small batch sizes may not fully utilize the GPU
   - Try increasing batch size with the PPO params in train_enemy_rl.py
   - Ensure the CUDA compute capability in CMakeLists.txt supports your GPU

4. **Environment Variable Issues**
   - If you're using conda, ensure you've activated the GPU environment:
     `conda activate ai-platform-gpu`

## Performance Tips

1. Always use `--headless` for faster training
2. For large training runs, use `--timesteps 1000000` or higher
3. Monitor GPU memory usage to optimize batch sizes
4. Consider using `nohup` for long-running training sessions:
   ```
   nohup python run_gpu_training.py --timesteps 1000000 --headless > training_log.txt &
   ```

## Understanding GPU Acceleration Components

The GPU acceleration in this project happens at two levels:

1. **PyTorch Neural Network Training**: The PPO algorithm uses PyTorch which can run on GPU
2. **C++ CUDA Custom Modules**: Physics calculations use custom CUDA kernels

For maximum performance benefits, ensure both components are working correctly.

## Monitoring GPU Usage

If you have `nvidia-smi` available on your system, you can also manually monitor GPU usage in a separate terminal:

```bash
# Check GPU status once
nvidia-smi

# Monitor GPU usage continuously (updates every 1 second)
watch -n 1 nvidia-smi
```

This will show GPU utilization, memory usage, and running processes.
