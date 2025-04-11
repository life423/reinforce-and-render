"""
Compatibility module for AI Platform Trainer spawner.

This module re-exports functions from the new location to maintain
backward compatibility with existing code during the refactoring process.
"""
import warnings

# Import from new location with prefixed names
from ai_platform_trainer.engine.physics.spawner import respawn_enemy as _respawn_enemy
from ai_platform_trainer.engine.physics.spawner import (
    respawn_enemy_at_position as _respawn_enemy_at_position,
)
from ai_platform_trainer.engine.physics.spawner import (
    respawn_enemy_with_fade_in as _respawn_enemy_with_fade_in,
)
from ai_platform_trainer.engine.physics.spawner import spawn_entities as _spawn_entities

# Re-export at module level
spawn_entities = _spawn_entities
respawn_enemy_at_position = _respawn_enemy_at_position
respawn_enemy = _respawn_enemy
respawn_enemy_with_fade_in = _respawn_enemy_with_fade_in

# Emit a deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.gameplay.spawner is deprecated. "
    "Use ai_platform_trainer.engine.physics.spawner instead.",
    DeprecationWarning,
    stacklevel=2
)