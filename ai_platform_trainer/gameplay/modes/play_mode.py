"""
Compatibility module for AI Platform Trainer play mode.

This module re-exports the PlayMode class from the new location to maintain
backward compatibility with existing code during the refactoring process.
"""
import warnings

# Import from new location with prefixed name
from ai_platform_trainer.engine.gameplay.modes.play_mode import PlayMode as _PlayMode

# Re-export at module level
PlayMode = _PlayMode

# Emit a deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.gameplay.modes.play_mode is deprecated. "
    "Use ai_platform_trainer.engine.gameplay.modes.play_mode instead.",
    DeprecationWarning,
    stacklevel=2
)