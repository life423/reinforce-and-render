"""
Compatibility module for AI Platform Trainer spawn utilities.

This module re-exports functions from the new location to maintain
backward compatibility with existing code during the refactoring process.
"""
import warnings

# Import from new location with prefixed names
from ai_platform_trainer.engine.gameplay.spawn_utils import (
    find_valid_spawn_position as _find_valid_spawn_position,
    generate_item_at_position as _generate_item_at_position,
    calculate_spawn_region as _calculate_spawn_region,
)

# Re-export at module level
find_valid_spawn_position = _find_valid_spawn_position
generate_item_at_position = _generate_item_at_position
calculate_spawn_region = _calculate_spawn_region

# Emit a deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.gameplay.spawn_utils is deprecated. "
    "Use ai_platform_trainer.engine.gameplay.spawn_utils instead.",
    DeprecationWarning,
    stacklevel=2
)