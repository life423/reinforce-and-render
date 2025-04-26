#!/usr/bin/env python
"""
Entry point script for training the enemy AI using reinforcement learning with GPU acceleration.

This script allows easy training of the enemy's RL model from the command line with CUDA support.
Usage: python train_enemy_rl_model.py [--timesteps 100000] [--headless]
"""
import argparse
import logging
import os
import sys
import time
import threading
import numpy as np
import matplotlib
import torch
from pathlib import Path
# matplotlib.use must be called before importing pyplot
# noqa: E402 - module level import not at top of file
matplotlib.use('Agg')  # Use non-interactive backend for headless operation

# Check for GPU support first
gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    logging.info(f"[SUCCESS] CUDA is available: {gpu_name}, CUDA version: {cuda_version}")
else:
    logging.warning("[WARNING] CUDA is NOT available! Training will use CPU only (slower).")

# Add the cpp directory to path to ensure extensions can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
cpp_dir = os.path.join(script_dir, 'ai_platform_trainer', 'cpp')
if cpp_dir not in sys.path:
    sys.path.append(cpp_dir)

# Try to import the GPU environment
try:
    from ai_platform_trainer.cpp.gpu_environment import make_env, HAS_GPU_ENV
    if HAS_GPU_ENV:
        logging.info("[SUCCESS] Successfully imported GPU-accelerated environment")
    else:
        logging.warning("[WARNING] GPU environment module found but not properly built with CUDA")
except ImportError:
    HAS_GPU_ENV = False
    logging.warning("[WARNING] Could not import GPU environment. C++ extension may not be built.")
    logging.warning("[INSTRUCTION] Run: cd ai_platform_trainer/cpp && python setup.py build_ext --inplace")

# Import regular environment as fallback and train_rl_agent
from ai_platform_trainer.ai_model.train_enemy_rl import train_rl_agent  # noqa: E402
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Global variables for monitoring
gpu_utilization = []
cpu_utilization = []
memory_usage = []
timestamps = []
stop_monitoring = False

def get_gpu_utilization():
    """Get current GPU utilization using nvidia-smi"""
    if not gpu_available:
        return 0.0
        
    try:
        import subprocess
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        return float(result.strip())
    except Exception as e:
        logging.error(f"Error getting GPU utilization: {e}")
        return 0.0

def get_cpu_utilization():
    """Get current CPU utilization"""
    import psutil
    return psutil.cpu_percent()

def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def monitor_resources():
    """Monitor GPU, CPU and memory usage"""
    global stop_monitoring, gpu_utilization, cpu_utilization, memory_usage, timestamps
    
    start_time = time.time()
    while not stop_monitoring:
        current_time = time.time() - start_time
        timestamps.append(current_time)
        
        # Get GPU utilization
        gpu_util = get_gpu_utilization()
        gpu_utilization.append(gpu_util)
        
        # Get CPU utilization
        cpu_util = get_cpu_utilization()
        cpu_utilization.append(cpu_util)
        
        # Get memory usage
        mem_usage = get_memory_usage()
        memory_usage.append(mem_usage)
        
        # Log every 10 seconds
        if int(current_time) % 10 == 0 and current_time > 0:
            logging.info(f"GPU: {gpu_util:.1f}%, CPU: {cpu_util:.1f}%, Memory: {mem_usage:.1f}MB")
        
        time.sleep(1.0)  # Sample every second

def plot_utilization(save_path):
    """Plot the utilization metrics"""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        
        # Plot GPU utilization
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, gpu_utilization, 'r-')
        plt.title('GPU Utilization')
        plt.ylabel('Utilization (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        
        # Plot CPU utilization
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, cpu_utilization, 'b-')
        plt.title('CPU Utilization')
        plt.ylabel('Utilization (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        
        # Plot memory usage
        plt.subplot(3, 1, 3)
        plt.plot(timestamps, memory_usage, 'g-')
        plt.title('Memory Usage')
        plt.ylabel('Usage (MB)')
        plt.xlabel('Time (s)')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_path, 'training_resource_usage.png')
        plt.savefig(plot_path)
        logging.info(f"Resource utilization plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to create utilization plot: {e}")


class GPUTrainingCallback(BaseCallback):
    """
    Custom callback for training that extends the checkpoint functionality
    and monitors GPU usage.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        log_path: str,
        name_prefix: str = "enemy_ppo_model"
    ):
        """
        Initialize the callback with saving parameters.

        Args:
            save_freq: Number of timesteps between checkpoints
            save_path: Directory to save models to
            log_path: Directory for logs
            name_prefix: Prefix for saved model files
        """
        super().__init__(verbose=1)
        self.save_freq = save_freq
        self.save_path = save_path
        self.log_path = log_path
        self.name_prefix = name_prefix
        self.best_reward = -float('inf')
        self.best_model_path = os.path.join(save_path, f"{name_prefix}_best")
        self.last_save_step = 0
        self.training_start_time = time.time()
        
        # GPU monitoring
        self.start_resource_monitoring()

    def _init_callback(self) -> None:
        """
        Initialize callback variables at the start of training.
        """
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Update metrics and save model when appropriate.

        Returns:
            Whether to continue training
        """
        # Check if we should save a model checkpoint
        if self.n_calls - self.last_save_step >= self.save_freq:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}")
            self.model.save(path)
            self.last_save_step = self.n_calls

            # Save the best model based on reward
            avg_reward = self._get_avg_reward()
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.model.save(self.best_model_path)
                logging.info(f"New best model with reward {avg_reward:.2f}")

        return True

    def _get_avg_reward(self) -> float:
        """
        Calculate average reward from episode info buffer.

        Returns:
            Average reward or 0 if no episodes completed
        """
        if len(self.model.ep_info_buffer) == 0:
            return 0.0

        return np.mean([ep['r'] for ep in self.model.ep_info_buffer])
        
    def start_resource_monitoring(self):
        """Start the thread that monitors GPU, CPU usage"""
        global stop_monitoring
        stop_monitoring = False
        self.monitor_thread = threading.Thread(target=monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_resource_monitoring(self):
        """Stop the monitoring thread and save the plot"""
        global stop_monitoring
        stop_monitoring = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        plot_utilization(self.log_path)


def train_gpu_rl_agent(
    total_timesteps: int = 500000,
    save_path: str = "models/enemy_rl",
    log_path: str = "logs/enemy_rl",
    headless: bool = True
):
    """
    Train a reinforcement learning agent for enemy behavior using GPU acceleration.

    Args:
        total_timesteps: Total number of timesteps to train for
        save_path: Directory to save model checkpoints
        log_path: Directory to save tensorboard logs
        headless: Whether to run without visualization (faster training)

    Returns:
        The trained PPO model or None if training fails
    """
    # Create directories if they don't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    try:
        if HAS_GPU_ENV:
            # Create the GPU-accelerated environment
            config = make_env()._default_config()
            
            # Customize the config as needed for enemy training
            config.enable_missile_avoidance = True
            config.missile_prediction_steps = 30
            
            # Create and wrap the environment
            env = DummyVecEnv([lambda: make_env(config)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            
            # Create the PPO model with explicit device selection
            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[
                    dict(
                        pi=[128, 128],  # Policy network
                        vf=[128, 128]   # Value function network
                    )
                ]
            )
            
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_path,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # Slightly higher entropy to encourage exploration
                policy_kwargs=policy_kwargs,
                device="cuda" if gpu_available else "cpu",
            )
            
            # Set up checkpoint callback with monitoring
            callback = GPUTrainingCallback(
                save_freq=10000,
                save_path=save_path,
                log_path=log_path,
                name_prefix="enemy_ppo_model"
            )
            
            # Train the agent
            logging.info(f"Starting GPU-accelerated PPO training for {total_timesteps} timesteps")
            logging.info(f"Using device: {model.device}")
            
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name="enemy_ppo_gpu",
                reset_num_timesteps=True
            )
            
            # Stop monitoring
            callback.stop_resource_monitoring()
            
            # Save the final model and normalized environment
            final_model_path = os.path.join(save_path, "final_model")
            model.save(final_model_path)
            env.save(os.path.join(save_path, "vec_normalize.pkl"))
            
            logging.info(f"Training completed. Model saved to {final_model_path}")
            
            # Check if GPU was effectively used
            if gpu_utilization:
                avg_gpu = np.mean(gpu_utilization)
                max_gpu = np.max(gpu_utilization)
                logging.info(f"Average GPU utilization: {avg_gpu:.1f}%")
                logging.info(f"Maximum GPU utilization: {max_gpu:.1f}%")
                
                if max_gpu < 10.0:
                    logging.warning("Low GPU utilization detected. The GPU may not have been effectively used.")
            
            return model
            
        else:
            # Fall back to CPU-based training
            logging.warning("Using CPU-based training as GPU environment is not available.")
            return train_rl_agent(
                total_timesteps=total_timesteps,
                save_path=save_path,
                log_path=log_path,
                headless=headless
            )

    except Exception as e:
        logging.error(f"Error during training: {e}")
        return None


def setup_logging():
    """Set up logging for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the enemy AI using reinforcement learning')
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
        '--visualize-interval',
        type=int,
        default=300,
        help='Interval in seconds between visualization updates (default: 300)'
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
    return parser.parse_args()


def ensure_directories(save_path, log_path):
    """Ensure the save and log directories exist."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point for the training script."""
    setup_logging()
    args = parse_args()

    # Create directories if they don't exist
    ensure_directories(args.save_path, args.log_path)

    # Log environment information
    logging.info(f"Starting enemy AI RL training for {args.timesteps} timesteps")
    logging.info(f"Model checkpoints will be saved to {args.save_path}")
    logging.info(f"Training logs will be saved to {args.log_path}")
    logging.info(f"Headless mode: {args.headless}")
    
    # GPU status information
    if gpu_available and not args.force_cpu:
        logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        if HAS_GPU_ENV:
            logging.info("GPU-accelerated environment is available and will be used")
        else:
            logging.warning("GPU-accelerated environment is NOT available.")
            logging.warning("Only PyTorch will use GPU, physics will run on CPU.")
    else:
        if args.force_cpu:
            logging.info("Forcing CPU usage as requested")
        else:
            logging.warning("No GPU detected, using CPU for training")

    # Train the model with GPU support if available
    model = train_gpu_rl_agent(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        log_path=args.log_path,
        headless=args.headless
    )

    if model is not None:
        logging.info("Training completed successfully!")
        final_path = os.path.join(args.save_path, 'final_model.zip')
        best_path = os.path.join(args.save_path, 'enemy_ppo_model_best.zip')
        logging.info(f"Final model saved to {final_path}")
        logging.info(f"Best model saved to {best_path}")

        # Show visualization info
        dashboard_path = os.path.join(args.log_path, "training_resource_usage.png")
        if os.path.exists(dashboard_path):
            logging.info(f"Training resource usage dashboard available at: {dashboard_path}")

        # Suggest next steps
        logging.info("\nNext steps:")
        logging.info("1. Run the game: python -m ai_platform_trainer.main")
        logging.info("2. View metrics: tensorboard --logdir=logs/enemy_rl")
        logging.info("3. Examine training visualizations in the logs directory")
    else:
        logging.error("Training failed! Check the logs for errors.")


if __name__ == "__main__":
    main()
