"""
Compatibility module for AI Platform Trainer spawn_utils.

This module re-exports functions from the new location to maintain
backward compatibility with existing code during the refactoring process.
"""
import warnings

# Import from new location with prefixed names
from ai_platform_trainer.engine.gameplay.spawn_utils import (
    find_valid_spawn_position as _find_valid_spawn_position,
    calculate_spawn_position as _calculate_spawn_position,
    create_enemy_spawn_positions as _create_enemy_spawn_positions
)

# Re-export at module level
find_valid_spawn_position = _find_valid_spawn_position
calculate_spawn_position = _calculate_spawn_position
create_enemy_spawn_positions = _create_enemy_spawn_positions

# Emit a deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.gameplay.spawn_utils is deprecated. "
    "Use ai_platform_trainer.engine.gameplay.spawn_utils instead.",
    DeprecationWarning,
    stacklevel=2
)