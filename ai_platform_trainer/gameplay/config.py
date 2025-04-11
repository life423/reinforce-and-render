"""
Compatibility module for AI Platform Trainer config.

This module re-exports config from its new location to maintain
backward compatibility with existing code during the refactoring process.
"""
import warnings

# Re-export config from its new location
from ai_platform_trainer.engine.gameplay.config import config

# Emit a deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.gameplay.config is deprecated. "
    "Use ai_platform_trainer.engine.gameplay.config instead.",
    DeprecationWarning,
    stacklevel=2
)