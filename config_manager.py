# config_manager.py
import json
import os

DEFAULT_SETTINGS = {"fullscreen": False, "width": 1280, "height": 720}


def load_settings(config_path: str = "settings.json") -> dict:
    """
    Loads settings from a JSON file. If the file doesn't exist,
    returns a default settings dict.
    """
    if os.path.isfile(config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    else:
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict, config_path: str = "settings.json") -> None:
    """
    Saves the settings dictionary to a JSON file.
    """
    with open(config_path, "w") as file:
        json.dump(settings, file, indent=4)
