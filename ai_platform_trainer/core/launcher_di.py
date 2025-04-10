"""
Dependency Injection Launcher Module (Adapter)

This is an adapter module that forwards to the canonical implementation
in engine/core/unified_launcher.py for backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.unified_launcher instead.
"""
import warnings
import os
from ai_platform_trainer.engine.core.unified_launcher import main, launch_dependency_injection

# Keep register_services function for backward compatibility
from ai_platform_trainer.core.service_locator import ServiceLocator
from ai_platform_trainer.core.config_manager import get_config_manager
from config_manager import load_settings, save_settings
import pygame
from ai_platform_trainer.gameplay.renderer_di import Renderer
from ai_platform_trainer.gameplay.input_handler import InputHandler
from ai_platform_trainer.entities.entity_factory import PlayEntityFactory, TrainingEntityFactory
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.display_manager import init_pygame_display
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.config import config

# Re-export the service registration functionality
# This is needed for the unified launcher to work with the DI system
from ai_platform_trainer.engine.core.launcher_di import (
    DisplayService, 
    SettingsService, 
    CollisionService, 
    MenuFactory, 
    register_services
)

# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.core.launcher_di is deprecated. "
    "Use ai_platform_trainer.engine.core.unified_launcher instead.",
    DeprecationWarning,
    stacklevel=2
)

# Override environment variable to use DI mode
os.environ["AI_PLATFORM_LAUNCHER_MODE"] = "DI"

# For backwards compatibility, provide the main function
__all__ = ["main", "register_services"]
