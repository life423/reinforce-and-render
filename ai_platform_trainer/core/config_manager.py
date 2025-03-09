# file: ai_platform_trainer/core/config_manager.py
"""
Configuration management system for the AI Platform Trainer.
Provides centralized access to configuration values.
"""
import json
import os
import logging
from typing import Any, Dict, Optional

# Default configuration values
DEFAULT_CONFIG = {
    "display": {
        "width": 800,
        "height": 600,
        "fullscreen": False,
        "window_title": "AI Platform Trainer",
        "frame_rate": 60,
    },
    "gameplay": {
        "respawn_delay": 1000,
        "player_step": 3,
        "player_size": 20,
        "enemy_step": 2,
        "enemy_size": 20,
        "missile_step": 4,
        "missile_size": 5,
    },
    "ai": {
        "missile_model_path": "models/missile_model.pth",
        "enemy_model_path": "models/enemy_ai_model.pth",
        "input_size": 5,
        "hidden_size": 64,
        "output_size": 2,
    },
    "paths": {
        "data_path": "data/raw/training_data.json",
    }
}


class ConfigManager:
    """
    Configuration manager for the AI Platform Trainer.
    Provides centralized access to configuration values from various sources.
    """
    
    def __init__(self, config_file: str = "config.json") -> None:
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        
        # Load configuration from default
        self.config.update(DEFAULT_CONFIG)
        
        # Load configuration from file
        self._load_from_file()
        
        logging.info("Configuration manager initialized")
    
    def _load_from_file(self) -> None:
        """Load configuration from file."""
        if not os.path.exists(self.config_file):
            logging.info(f"Configuration file {self.config_file} not found. Using defaults.")
            return
        
        try:
            with open(self.config_file, "r") as f:
                file_config = json.load(f)
                
            # Update config with values from file
            # This will recursively update nested dictionaries
            self._deep_update(self.config, file_config)
            logging.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logging.error(f"Error loading configuration from {self.config_file}: {e}")
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary.
        
        Args:
            target: The dictionary to update
            source: The dictionary to update from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # If both target and source have a dict at this key, recursively update
                self._deep_update(target[key], value)
            else:
                # Otherwise, just update the value
                target[key] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Error saving configuration to {self.config_file}: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "display.width")
            default: Default value to return if the key path is not found
            
        Returns:
            The configuration value, or the default value if not found
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value by key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "display.width")
            value: The value to set
        """
        keys = key_path.split(".")
        config = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: str = "config.json") -> ConfigManager:
    """
    Get the singleton ConfigManager instance.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        The ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager
