"""
Unified Launcher Module for AI Platform Trainer

This module provides a consolidated entry point for the game with support for
different game initialization methods (dependency injection, standard, or state machine).
It selects the appropriate launcher based on settings and provides fallback mechanisms.
"""
import logging
import os
import sys
import traceback
from enum import Enum, auto

# from typing import Any, Dict, Optional

try:
    from ai_platform_trainer.core.logging_config import setup_logging

    # from ai_platform_trainer.core.service_locator import ServiceLocator
    from ai_platform_trainer.engine.core.game import Game as StandardGame
    from ai_platform_trainer.engine.core.game_di import Game as DIGame
    from ai_platform_trainer.engine.core.game_refactored import Game as StateMachineGame
except ImportError as e:
    print(f"Critical import error: {e}")
    print("Cannot initialize launcher system.")
    sys.exit(1)


class LauncherMode(Enum):
    """Enum for different launcher modes."""
    STANDARD = auto()
    DEPENDENCY_INJECTION = auto()
    STATE_MACHINE = auto()


def get_launcher_mode_from_settings() -> LauncherMode:
    """
    Determine which launcher mode to use based on settings.
    
    Returns:
        LauncherMode: The launcher mode to use
    """
    try:
        env_mode = os.environ.get("AI_PLATFORM_LAUNCHER_MODE", "").upper()
        if env_mode == "STANDARD":
            return LauncherMode.STANDARD
        elif env_mode == "DI":
            return LauncherMode.DEPENDENCY_INJECTION
        elif env_mode == "STATE_MACHINE":
            return LauncherMode.STATE_MACHINE
        
        try:
            from config_manager import load_settings
            settings = load_settings("settings.json")
            if settings and "launcher_mode" in settings:
                mode = settings["launcher_mode"].upper()
                if mode == "STANDARD":
                    return LauncherMode.STANDARD
                elif mode == "DI":
                    return LauncherMode.DEPENDENCY_INJECTION
                elif mode == "STATE_MACHINE":
                    return LauncherMode.STATE_MACHINE
        except Exception as settings_error:
            logging.warning(f"Could not load settings: {settings_error}")
            
        return LauncherMode.DEPENDENCY_INJECTION
    except Exception as e:
        logging.error(f"Error determining launcher mode: {e}")
        return LauncherMode.STANDARD


def launch_standard() -> None:
    """
    Launch the game using the standard launcher.
    """
    try:
        game = StandardGame()
        game.run()
    except Exception as e:
        logging.error(f"Error in standard launcher: {e}")
        logging.debug(traceback.format_exc())
        raise


def launch_dependency_injection() -> None:
    """
    Launch the game using the dependency injection launcher.
    
    Sets up all services and registers them with the service locator before
    creating and running the game.
    """
    try:
        from ai_platform_trainer.core.launcher_di import register_services
        
        register_services()
        
        game = DIGame()
        game.run()
    except Exception as e:
        logging.error(f"Error in DI launcher: {e}")
        logging.debug(traceback.format_exc())
        logging.warning("Falling back to standard launcher...")
        launch_standard()


def launch_state_machine() -> None:
    """
    Launch the game using the state machine launcher.
    """
    try:
        game = StateMachineGame()
        game.run()
    except Exception as e:
        logging.error(f"Error in state machine launcher: {e}")
        logging.debug(traceback.format_exc())
        logging.warning("Falling back to DI launcher...")
        try:
            launch_dependency_injection()
        except Exception:
            logging.warning("Falling back to standard launcher...")
            launch_standard()


def main() -> None:
    """
    Main entry point for the AI Platform Trainer.
    
    Selects the appropriate launcher based on settings and runs the game.
    Provides fallback mechanisms if the selected launcher fails.
    """
    setup_logging()
    
    logging.info("Starting AI Platform Trainer")
    
    try:
        mode = get_launcher_mode_from_settings()
        
        logging.info(f"Using launcher mode: {mode.name}")
        
        if mode == LauncherMode.STANDARD:
            launch_standard()
        elif mode == LauncherMode.DEPENDENCY_INJECTION:
            launch_dependency_injection()
        elif mode == LauncherMode.STATE_MACHINE:
            launch_state_machine()
        else:
            logging.error(f"Unknown launcher mode: {mode}")
            raise ValueError(f"Unknown launcher mode: {mode}")
            
    except Exception as e:
        logging.critical(f"Fatal error in game launcher: {e}")
        logging.debug(traceback.format_exc())
        raise
    finally:
        logging.info("Exiting AI Platform Trainer")


if __name__ == "__main__":
    main()
