"""
Configuration Management Module (Adapter)

This is an adapter module that forwards to the canonical implementation
in engine/core/config_manager.py for backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.config_manager instead.
"""
import warnings
import json
import os
from typing import Dict, Any


# Re-export the get_config_manager function from engine/core
from ai_platform_trainer.engine.core.config_manager import get_config_manager  # noqa


# Simple settings load function for backward compatibility
DEFAULT_SETTINGS = {
    "fullscreen": False,
    "width": 1280,
    "height": 720,
}


def load_settings(config_path: str = "settings.json") -> Dict[str, Any]:
    """
    Load settings from a JSON file. If the file doesn't exist, returns a default dict.
    """
    if os.path.isfile(config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any], config_path: str = "settings.json") -> None:
    """
    Save the settings dictionary to a JSON file.
    """
    with open(config_path, "w") as file:
        json.dump(settings, file, indent=4)


# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.core.config_manager is deprecated. "
    "Use ai_platform_trainer.engine.core.config_manager instead.",
    DeprecationWarning,
    stacklevel=2
)
