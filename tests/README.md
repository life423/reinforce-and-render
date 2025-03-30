# Test Suite for AI Platform Trainer

This directory contains tests for the AI Platform Trainer project, organized into different categories to ensure comprehensive coverage of the game's functionality.

## Directory Structure

```
tests/
│
├── unit/                  # Unit tests for isolated components
│   ├── entities/          # Tests for game entities
│   │   ├── behaviors/     # Tests for entity behaviors
│   │   └── components/    # Tests for entity components
│   ├── engine/            # Tests for engine components
│   │   ├── core/          # Tests for core engine functionality
│   │   ├── physics/       # Tests for physics engine
│   │   └── rendering/     # Tests for rendering engine
│   └── ai/                # Tests for AI models
│       ├── models/        # Tests for model definitions
│       └── training/      # Tests for training functionality
│
├── integration/           # Integration tests between components
│   ├── test_game_mechanics.py  # Tests for game mechanics
│   └── test_training_pipeline.py  # Tests for training pipeline
│
└── performance/           # Tests for performance characteristics
```

## Key Test Files

### Unit Tests

- **Enemy AI Controller** (`unit/entities/behaviors/test_enemy_ai_controller.py`): Tests the enemy movement logic, focusing on:
  - Stuck detection and recovery
  - Different movement strategies
  - Neural network and reinforcement learning integration
  - Movement normalization

- **Missile Controller** (`unit/entities/behaviors/test_missile_ai_controller.py`): Tests missile guidance, including:
  - Target tracking
  - Blending between AI model output and direct targeting
  - Handling multiple missiles
  - Turn rate constraints

- **Player** (`unit/entities/components/test_player.py`): Tests player functionality, including:
  - Input handling
  - Missile firing
  - Movement patterns (for training mode)
  - Screen wrapping

- **Collision Detection** (`unit/engine/physics/test_collisions.py`): Tests collision handling between:
  - Missiles and enemies
  - Multiple missiles
  - Invisible vs. visible entities

### Integration Tests

- **Game Mechanics** (`integration/test_game_mechanics.py`): Tests interactions between:
  - Player movement and input
  - Enemy AI following the player
  - Missile firing and collision detection
  - Game loop mechanics

## Running the Tests

To run all tests:
```bash
# Windows
run_tests.bat

# Linux/macOS
./run_tests.sh
```

To run specific test categories:
```bash
# Run only unit tests
python -m pytest tests/unit/

# Run only tests for enemy AI
python -m pytest tests/unit/entities/behaviors/test_enemy_ai_controller.py

# Run with coverage
python -m pytest --cov=ai_platform_trainer tests/
```

## Adding New Tests

When adding new tests:
1. Place them in the appropriate directory
2. Follow the naming convention `test_*.py` for files and `test_*` for test methods
3. Use pytest fixtures for common setup
4. Mock external dependencies like pygame
5. Reference the testing guide for best practices: `docs/testing_guide.md`

## CI/CD Integration

These tests can be integrated with continuous integration systems to ensure that new changes don't introduce regressions.

An example GitHub Actions workflow would look like:

```yaml
name: Run Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
