# AI Platform Trainer - Launcher System

## Overview

The AI Platform Trainer game now uses a unified launcher system that provides a consistent entry point while supporting multiple initialization methods. This document explains how the launcher system works and how to configure it.

## Launcher Modes

The game supports three launcher modes:

1. **Standard Mode** (`STANDARD`)
   - Basic initialization with minimal dependencies
   - Fastest startup time
   - Limited configuration options

2. **Dependency Injection Mode** (`DI`)
   - Uses Service Locator pattern for component registration
   - More flexible and extensible
   - Better testability
   - **Default mode**

3. **State Machine Mode** (`STATE_MACHINE`)
   - Uses a state machine architecture for game flow
   - Better separation of game states
   - Easier to add new game states

## Configuration

You can specify which launcher mode to use in two ways:

### 1. Environment Variable

Set the `AI_PLATFORM_LAUNCHER_MODE` environment variable to one of:
- `STANDARD`
- `DI`
- `STATE_MACHINE`

Example:
```bash
# Windows
set AI_PLATFORM_LAUNCHER_MODE=STATE_MACHINE
python -m ai_platform_trainer.main

# Linux/Mac
export AI_PLATFORM_LAUNCHER_MODE=STATE_MACHINE
python -m ai_platform_trainer.main
```

### 2. Settings File

Modify the `settings.json` file to include the launcher mode:

```json
{
  "fullscreen": true,
  "launcher_mode": "DI"
}
```

Valid values for `launcher_mode` are:
- `STANDARD`
- `DI`
- `STATE_MACHINE`

## Fallback Behavior

The launcher system includes fallback mechanisms to ensure the game can start even if the preferred launcher mode fails:

1. If the `STATE_MACHINE` mode fails, it falls back to `DI` mode
2. If the `DI` mode fails, it falls back to `STANDARD` mode
3. If all modes fail, the game will exit with an error message

## Developer Information

### Code Organization

The launcher system is organized as follows:

- `ai_platform_trainer/main.py` - Main entry point
- `ai_platform_trainer/engine/core/unified_launcher.py` - Unified launcher implementation
- `ai_platform_trainer/core/launcher.py` - Standard launcher (legacy)
- `ai_platform_trainer/core/launcher_di.py` - DI launcher (legacy)
- `ai_platform_trainer/core/launcher_refactored.py` - State machine launcher (legacy)

### Extending the Launcher

To add a new launcher mode:

1. Add a new mode to the `LauncherMode` enum in `unified_launcher.py`
2. Create a launcher function (similar to `launch_standard()`)
3. Update the `main()` function to handle the new mode
4. Update the `get_launcher_mode_from_settings()` function to recognize the new mode
5. Update this documentation

### Logging

The launcher system logs detailed information about launcher mode selection and any errors that occur. Check the logs at `logs/gameplay/game.log` for troubleshooting information.

## Troubleshooting

### Game Crashes on Startup

1. Check the log file at `logs/gameplay/game.log` for error messages
2. Try running with a different launcher mode using the environment variable
3. Verify that all required dependencies are installed

### Unexpected Behavior with DI or State Machine

Try falling back to the standard launcher mode:

```bash
set AI_PLATFORM_LAUNCHER_MODE=STANDARD
python -m ai_platform_trainer.main
```

If the game works correctly in standard mode, there may be an issue with the DI or state machine configuration.
