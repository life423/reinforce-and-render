# AI Platform Trainer

A game environment for training and evaluating AI agents through reinforcement learning.

## Overview

AI Platform Trainer provides a framework for training AI-controlled entities in a simulated game environment. The platform features:

- Neural network-based enemy AI
- Reinforcement learning integration
- Real-time visualization of training progress
- Custom sprite rendering system

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-platform-trainer.git
cd ai-platform-trainer
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Generate sprites:
```bash
python generate_sprites.py
```

### Running the Game

To start the game in play mode:
```bash
python -m ai_platform_trainer.main
```

## Training the AI

### Training Neural Network Model

To train the traditional neural network model:
```bash
python -m ai_platform_trainer.ai_model.train_missile_model
```

### Training Reinforcement Learning Model

To train the RL model:
```bash
python train_enemy_rl_model.py --timesteps 100000 --headless
```

Options:
- `--timesteps`: Number of training steps (default: 500000)
- `--headless`: Run without visualization for faster training
- `--save-path`: Directory to save model checkpoints (default: models/enemy_rl)
- `--log-path`: Directory to save logs and visualizations (default: logs/enemy_rl)

### Monitoring Training Progress

Training visualizations are saved to the logs directory and include:
- Learning curves
- Reward plots
- Behavioral metrics
- Performance dashboards

## Code Structure

- `ai_platform_trainer/`
  - `ai_model/`: Neural network and RL model definitions
    - `training_monitor.py`: Training visualization dashboard
    - `enemy_rl_agent.py`: RL environment for enemy training
    - `train_enemy_rl.py`: RL training implementation
  - `core/`: Core engine components 
  - `entities/`: Game entity definitions
  - `gameplay/`: Game logic and mechanics
    - `ai/`: AI controllers
      - `enemy_ai_controller.py`: Enemy movement AI
    - `renderer.py`: Sprite-based rendering
  - `utils/`: Utility functions
    - `sprite_manager.py`: Sprite loading and rendering

## Features

### Sprite Rendering System

The platform now includes a sprite rendering system that:
- Loads PNG sprite assets from the assets/sprites directory
- Falls back to procedurally generated sprites if assets aren't available
- Supports animations and particle effects
- Handles entity rotation and scaling

### Advanced Enemy AI

The enemy AI has been enhanced to prevent freezing behavior:
- Multi-strategy approach using both neural networks and reinforcement learning
- Position history tracking to detect and escape from stuck conditions
- Movement smoothing for more natural behavior
- Automatic fallback behaviors when primary AI strategies fail

### Training Visualization

The training monitoring system provides:
- Real-time training metrics
- Visual dashboards of agent performance
- Animation of behavioral patterns
- Exportable reports and charts

## Troubleshooting

If the enemy appears to freeze or behave erratically:
1. Check that the model files exist in the models directory
2. Try generating a new reinforcement learning model with `python train_enemy_rl_model.py`
3. Ensure pygame is properly installed and rendering correctly
4. Check the logs for any error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.
