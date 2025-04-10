# AI Platform Trainer

A game environment for training and evaluating AI agents through reinforcement learning with PyTorch neural networks and C++/CUDA acceleration.

## Overview

AI Platform Trainer is an enterprise-grade framework for training AI-controlled entities in a simulated game environment. The platform features:

- Neural network-based AI with PyTorch
- Reinforcement learning integration with stable-baselines3
- C++/CUDA acceleration for physics computations
- CPU and GPU support with automatic detection
- Comprehensive logging and error handling
- Containerized deployment options

## Project Structure

The codebase follows a clean, modular structure:

```
ai_platform_trainer/
├── src/                       # All source code
│   └── ai_platform_trainer/   # Main package
│       ├── core/              # Core engine components
│       ├── ml/                # Machine learning components
│       │   ├── models/        # Neural network definitions
│       │   ├── training/      # Training pipelines
│       │   ├── rl/            # Reinforcement learning
│       │   └── inference/     # Model inference
│       ├── physics/           # Physics engine (with C++ bindings)
│       ├── entities/          # Game entities
│       ├── rendering/         # Visualization and rendering
│       └── utils/             # Utility functions
├── tests/                     # Test suite
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── assets/                    # Game assets
└── deployment/                # Deployment configurations
    ├── docker/                # Docker files
    └── ci/                    # CI/CD configuration
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- For GPU acceleration: CUDA Toolkit 11.x and compatible GPU

### Installation

#### Option 1: Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-platform-trainer.git
cd ai-platform-trainer
```

2. Set up a virtual environment:
```bash
# CPU-only environment
conda env create -f environment-cpu.yml
conda activate ai-platform-cpu

# OR for GPU support
conda env create -f environment-gpu.yml
conda activate ai-platform-gpu
```

3. Install the package in development mode:
```bash
pip install -e .
```

#### Option 2: Docker Installation

We provide Docker containers for both CPU and GPU environments:

```bash
# CPU environment
cd deployment/docker
docker-compose up dev-cpu

# GPU environment (requires nvidia-docker)
docker-compose up dev-gpu
```

### Environment Setup Verification

Verify your environment setup with:

```bash
python -m src.ai_platform_trainer.utils.environment
```

This will show the detected environment configuration, including CUDA availability.

## Usage

### Running the Game

To start the game in play mode:
```bash
python -m src.ai_platform_trainer.main
```

### Training Models

#### Training Neural Network Model

To train the traditional neural network model:
```bash
python -m src.ai_platform_trainer.ml.training.train_missile_model
```

#### Training Reinforcement Learning Model

To train the RL model:
```bash
python -m src.ai_platform_trainer.ml.rl.train_enemy_rl
```

Options:
- `--timesteps`: Number of training steps (default: 500000)
- `--headless`: Run without visualization for faster training
- `--save-path`: Directory to save model checkpoints (default: models/enemy_rl)
- `--log-path`: Directory to save logs (default: logs/enemy_rl)

## Development

### Code Quality

We use several tools to ensure code quality:

1. **Formatting and Linting**: Run pre-commit hooks on your changes:
```bash
pre-commit install  # Set up the git hooks
pre-commit run --all-files  # Manually run on all files
```

2. **Type Checking**: The codebase uses mypy for static type checking:
```bash
mypy src/
```

3. **Testing**: Run the test suite:
```bash
pytest tests/
```

### Docker Development

For containerized development:

```bash
cd deployment/docker
docker-compose up dev-cpu  # or dev-gpu
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow for CI/CD:

- **Linting**: Enforces code style and quality standards
- **Testing**: Runs the test suite on CPU environments
- **Building**: Creates distribution packages
- **GPU Testing**: (Optional, requires self-hosted runners)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
