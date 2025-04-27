# PhysX: AI Training Platform

A modern Python platform for training AI agents in physics-based game environments. PhysX combines a Pygame-powered game engine with PyTorch machine learning capabilities to create an ideal environment for AI research and education.

## 🚀 Features

- **Game Engine:** Built on Pygame with physics simulation capabilities
- **AI Training:** Reinforcement learning and supervised learning frameworks
- **GPU Acceleration:** CUDA support for faster model training
- **Multiple Demos:** Ready-to-run examples to get started quickly
- **Extensible Design:** Modular architecture that's easy to customize

## 🔧 Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment tool (venv, conda, etc.)
- GPU with CUDA support (optional, but recommended for AI training)

### Installation

1. **Create and activate a virtual environment:**

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**

   ```bash
   pip install -e .
   ```

4. **Fetch game assets:**

   ```bash
   python scripts/fetch_assets.py
   ```

## 🎮 Running the Platform

### Core Game

Run the main game engine with the following command:

```bash
ai-trainer
```

### Pygame Zero Demo

Try the simplified Pygame Zero demo:

```bash
pgzrun demos/pgz_demo/game_zero.py
```

## 🧠 AI Training

PhysX supports multiple AI training paradigms:

### Reinforcement Learning

- Vectorized environments in Pymunk
- Policy networks (MLP) trained via PPO
- GPU acceleration with PyTorch
- Real-time inference driving entity behavior

### Supervised Learning

- Custom data loaders for training datasets
- Model architecture examples for game environments
- Demonstration capabilities to visualize learning

## 📂 Project Structure

```
.
├── ai_platform_trainer/       # Core package
│   ├── __main__.py            # 'ai-trainer' entry point
│   ├── agents/                # RL policies & training loops
│   ├── core/                  # Config, logging, core utilities
│   ├── cpp/                   # CUDA/C++ acceleration code
│   ├── engine/                # Game engine components
│   │   ├── entities/          # Game entities (player, enemies, etc.)
│   │   ├── collision.py       # Physics collision handling
│   │   ├── display_manager.py # Display and window management
│   │   ├── game.py            # Main game loop
│   │   ├── input_handler.py   # User input processing
│   │   ├── menu.py            # UI menu system
│   │   └── renderer.py        # Graphics rendering
│   └── supervised/            # Supervised learning components
├── assets/sprites/            # Game assets (CC0 licensed)
├── demos/                     # Example implementations
├── scripts/                   # Utility scripts
└── [configuration files]      # Project configuration
```

## 🔄 How It Works

### Game Loop

1. **Input & AI Actions:** Process user input and AI agent decisions
2. **Physics & Updates:** Update entity positions and physics simulations
3. **Rendering:** Draw the current game state at 60 FPS

### AI Agent Integration

1. **Environment Observation:** Capture game state for AI input
2. **Model Inference:** Process state through neural networks
3. **Action Selection:** Determine optimal actions based on policy
4. **Environment Interaction:** Execute actions and update game state

## 🛠️ Extending the Platform

### Custom Entities

Create new entity types in `ai_platform_trainer/engine/entities/` by inheriting from the base `Entity` class.

### New Physics Demos

Add custom physics simulations by creating new components that use the collision system.

### AI Model Experiments

Implement custom neural network architectures in the supervised or reinforcement learning modules.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

Email: drew@drewclark.io  
GitHub: @life423
