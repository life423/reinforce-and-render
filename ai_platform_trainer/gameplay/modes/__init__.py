"""
Compatibility module for AI Platform Trainer game modes.

This module provides backward compatibility for imports from the old location.
"""
import warnings

# Emit a deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.gameplay.modes is deprecated. "
    "Use ai_platform_trainer.engine.gameplay.modes instead.",
    DeprecationWarning,
    stacklevel=2
)