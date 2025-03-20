"""
Training script for the enemy AI using Proximal Policy Optimization (PPO).

This module provides functionality to train an RL agent for enemy behavior
using the Stable Baselines3 PPO implementation.
"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ai_platform_trainer.ai_model.enemy_rl_agent import EnemyGameEnv
from ai_platform_trainer.gameplay.game import Game


class TrainingCallback(CheckpointCallback):
    """
    Custom callback for training that extends the checkpoint functionality.
    """
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "enemy_ppo_model"):
        """
        Initialize the callback with saving parameters.
        
        Args:
            save_freq: Number of timesteps between checkpoints
            save_path: Directory to save models to
            name_prefix: Prefix for saved model files
        """
        super().__init__(save_freq, save_path, name_prefix)
        self.best_reward = -float('inf')
        self.best_model_path = os.path.join(save_path, f"{name_prefix}_best")
    
    def _on_step(self) -> bool:
        """
        Check if we should save a model checkpoint on this step.
        
        Returns:
            Whether to continue training
        """
        # Call the parent _on_step for regular checkpoints
        super()._on_step()
        
        # Add custom logic to save the best model by reward
        if len(self.model.ep_info_buffer) > 0:
            avg_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                logging.info(f"New best model with reward {avg_reward:.2f}")
                self.model.save(self.best_model_path)
        
        return True


def train_rl_agent(
    total_timesteps: int = 500000,
    save_path: str = "models/enemy_rl",
    log_path: str = "logs/enemy_rl",
    headless: bool = True
) -> Optional[PPO]:
    """
    Train a reinforcement learning agent for enemy behavior.
    
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
        # Create a dedicated game instance for training
        game = Game()
        game.start_game("play")  # Initialize in play mode
        game.training_mode = True  # Special flag to indicate RL training
        
        if headless:
            # Replace rendering with dummy operations for headless training
            game.renderer.render = lambda *args, **kwargs: None
            
        # Create and wrap the environment
        def make_env():
            return EnemyGameEnv(game_instance=game)
        
        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Create the model with hyperparameters tuned for this task
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
            policy_kwargs=dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])]
            )
        )
        
        # Set up checkpointing and callbacks
        callbacks = [
            TrainingCallback(
                save_freq=10000,
                save_path=save_path,
                name_prefix="enemy_ppo_model"
            )
        ]
        
        # Train the agent
        logging.info(f"Starting PPO training for {total_timesteps} timesteps")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="enemy_ppo",
            reset_num_timesteps=True
        )
        
        # Save the final model and normalized environment
        final_model_path = os.path.join(save_path, "final_model")
        model.save(final_model_path)
        env.save(os.path.join(save_path, "vec_normalize.pkl"))
        
        logging.info(f"Training completed. Model saved to {final_model_path}")
        
        return model
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return None


def load_rl_model(
    model_path: str = "models/enemy_rl/final_model.zip",
    normalize_path: str = "models/enemy_rl/vec_normalize.pkl"
) -> Optional[PPO]:
    """
    Load a trained PPO model with normalization stats.
    
    Args:
        model_path: Path to the saved model
        normalize_path: Path to saved normalization statistics
        
    Returns:
        The loaded model or None if loading fails
    """
    try:
        # Create dummy environment
        def make_env():
            return EnemyGameEnv(game_instance=None)  # No game instance for loading
        
        dummy_env = DummyVecEnv([make_env])
        
        # Load normalization stats if available
        if os.path.exists(normalize_path):
            env = VecNormalize.load(normalize_path, dummy_env)
            # Don't update normalization statistics during evaluation
            env.training = False
            env.norm_reward = False
        else:
            env = dummy_env
            logging.warning(f"No normalization stats found at {normalize_path}")
            
        # Load the model
        model = PPO.load(model_path, env=env)
        logging.info(f"Successfully loaded model from {model_path}")
        
        return model
    
    except Exception as e:
        logging.error(f"Failed to load RL model: {e}")
        return None


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train the agent
    model = train_rl_agent(
        total_timesteps=100000,  # Lower for testing, increase for better results
        save_path="models/enemy_rl",
        log_path="logs/enemy_rl",
        headless=True
    )
    
    if model is not None:
        logging.info("Training successful")
