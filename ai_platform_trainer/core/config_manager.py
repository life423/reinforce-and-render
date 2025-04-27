import json
from pathlib import Path

_config = None

def load_settings(path: str = "config.json") -> dict:
    p = Path(path)
    if p.is_file():
        return json.loads(p.read_text())
    return {}

def save_settings(settings: dict, path: str = "config.json") -> None:
    Path(path).write_text(json.dumps(settings, indent=4))

def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_settings()
    return _config
