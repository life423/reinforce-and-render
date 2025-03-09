# AI Platform Trainer: Module Relationships

Since the automated dependency graph generation failed (due to missing Graphviz/dot tool), this document provides a manually created overview of module relationships in the codebase.

## Main Entry Point

```
ai_platform_trainer/main.py
  ├── imports core/launcher.py
  ├── imports gameplay/game.py
  └── orchestrates game initialization and execution
```

## Core Module Relationships

```
core/launcher.py
  ├── imports core/logging_config.py
  └── used by main.py for application initialization

core/data_logger.py
  ├── used by gameplay modules to record training data
  └── writes to data/raw/training_data.json
```

## Gameplay Module Relationships

```
gameplay/game.py
  ├── imports gameplay/renderer.py
  ├── imports gameplay/collisions.py
  ├── imports gameplay/config.py
  ├── imports entities/(player, enemy, missile).py
  ├── imports gameplay/modes/(play_mode, training_mode).py
  └── central orchestration point for gameplay

gameplay/renderer.py
  ├── imported by gameplay/game.py
  └── handles visual representation of game state

gameplay/config.py
  ├── imported by multiple gameplay modules
  └── contains game configuration parameters

gameplay/modes/play_mode.py
  ├── imports gameplay/game.py
  ├── imports entities/player_play.py
  └── implements regular gameplay mode

gameplay/modes/training_mode.py
  ├── imports gameplay/game.py
  ├── imports entities/player_training.py
  ├── imports core/data_logger.py
  └── implements AI training-focused mode
```

## Entity Module Relationships

```
entities/player.py (base class)
  ├── extended by entities/player_play.py
  └── extended by entities/player_training.py

entities/enemy.py (base class)
  ├── extended by entities/enemy_play.py
  ├── extended by entities/enemy_training.py
  └── may use entities/ai/ behaviors

entities/missile.py
  ├── used by player and enemy entities
  └── potentially influenced by gameplay/missile_ai_controller.py
```

## AI Model Module Relationships

```
ai_model/missile_dataset.py
  ├── imports torch
  ├── reads from data/raw/training_data.json
  └── used by ai_model/train_missile_model.py

ai_model/simple_missile_model.py
  ├── imports torch
  └── defines neural network architecture

ai_model/train_missile_model.py
  ├── imports ai_model/missile_dataset.py
  ├── imports ai_model/simple_missile_model.py
  └── trains and saves model to models/missile_model.pth

ai_model/model_definition/enemy_movement_model.py
  ├── imports torch
  └── defines neural network architecture for enemy movement
```

## AI Controller Relationships

```
gameplay/missile_ai_controller.py
  ├── imports torch
  ├── loads models/missile_model.pth
  └── used by gameplay/game.py for missile trajectory control

gameplay/ai/enemy_ai_controller.py
  ├── imports torch
  ├── may load models/enemy_ai_model.pth
  └── used by enemy entities for behavior control
```

## Key Circular Dependencies

1. **Game and Modes**:
   - gameplay/game.py ↔ gameplay/modes/play_mode.py
   - gameplay/game.py ↔ gameplay/modes/training_mode.py

2. **Entities and Controllers**:
   - entities/missile.py ↔ gameplay/missile_ai_controller.py
   - entities/enemy.py ↔ gameplay/ai/enemy_ai_controller.py

## File I/O Dependencies

1. **Training Data**:
   - core/data_logger.py → writes → data/raw/training_data.json
   - ai_model/missile_dataset.py → reads → data/raw/training_data.json

2. **Model Files**:
   - ai_model/train_missile_model.py → writes → models/missile_model.pth
   - gameplay/missile_ai_controller.py → reads → models/missile_model.pth

## Architectural Observations

1. **Tight Coupling**: Many modules directly import concrete implementations rather than interfaces, creating tight coupling.

2. **Circular Dependencies**: Several circular import dependencies exist, indicating potential design issues.

3. **Configuration Scattering**: Configuration parameters are spread across multiple files rather than centralized.

4. **Mixed Responsibilities**: Some modules handle both game logic and AI-related tasks, violating separation of concerns.
