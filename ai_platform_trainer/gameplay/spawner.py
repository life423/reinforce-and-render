"""
Compatibility module for AI Platform Trainer spawner.

This module re-exports functions from the new location to maintain
backward compatibility with existing code during the refactoring process.
"""
import warnings

# Re-export all relevant functions from the new location
from ai_platform_trainer.engine.physics.spawner import (
    spawn_entities,
    respawn_enemy_at_position,
    respawn_enemy,
    respawn_enemy_with_fade_in
)

# Emit a deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.gameplay.spawner is deprecated. "
    "Use ai_platform_trainer.engine.physics.spawner instead.",
    DeprecationWarning,
    stacklevel=2
)