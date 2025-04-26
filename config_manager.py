"""
Configuration Management Module (Root-level Adapter)

This is an adapter module that forwards to the canonical implementation
in ai_platform_trainer.engine.core.config_manager for backward compatibility.

It also imports functions from ai_platform_trainer.core.config_manager for
backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.config_manager instead.
"""
import json
import os
import warnings
from typing import Any, Dict

# Import the canonical modules
import ai_platform_trainer.engine.core.config_manager as engine_config_manager
from ai_platform_trainer.core.config_manager import load_settings, save_settings

# Default settings for backward compatibility
DEFAULT_SETTINGS = {
    "fullscreen": False,
    "width": 1280,
    "height": 720,
}

# Re-export all public attributes from engine config manager
__all__ = [name for name in dir(engine_config_manager) if not name.startswith('_')]
globals().update({name: getattr(engine_config_manager, name) for name in __all__})

# Make sure load_settings and save_settings are in __all__
if 'load_settings' not in __all__:
    __all__.append('load_settings')
if 'save_settings' not in __all__:
    __all__.append('save_settings')

# Add deprecation warning
warnings.warn(
    "Importing from root-level config_manager is deprecated. "
    "Use ai_platform_trainer.engine.core.config_manager instead.",
    DeprecationWarning,
    stacklevel=2
)