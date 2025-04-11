"""
Compatibility module for AI Platform Trainer training mode.

This module re-exports the TrainingMode class from the new location to maintain
backward compatibility with existing code during the refactoring process.
"""
import warnings

# Import from new location with prefixed name
from ai_platform_trainer.engine.gameplay.modes.training_mode import TrainingMode as _TrainingMode

# Re-export at module level
TrainingMode = _TrainingMode

# Emit a deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.gameplay.modes.training_mode is deprecated. "
    "Use ai_platform_trainer.engine.gameplay.modes.training_mode instead.",
    DeprecationWarning,
    stacklevel=2
)