#!/usr/bin/env python
"""
Entry point script for training the enemy AI using reinforcement learning.

This script allows easy training of the enemy's RL model from the command line.
Usage: python train_enemy_rl_model.py [--timesteps 100000] [--headless]
"""
import argparse
import logging
import os
from pathlib import Path

from ai_platform_trainer.ai_model.train_enemy_rl import train_rl_agent


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
    
    logging.info(f"Starting enemy AI RL training for {args.timesteps} timesteps")
    logging.info(f"Model will be saved to {args.save_path}")
    logging.info(f"Logs will be saved to {args.log_path}")
    logging.info(f"Headless mode: {args.headless}")
    
    # Train the model
    model = train_rl_agent(
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
        
        # Suggest next steps
        logging.info("\nNext steps:")
        logging.info("1. Run the game: python -m ai_platform_trainer.main")
        logging.info("2. View metrics: tensorboard --logdir=logs/enemy_rl")
    else:
        logging.error("Training failed! Check the logs for errors.")


if __name__ == "__main__":
    main()
