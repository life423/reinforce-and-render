#!/usr/bin/env python
"""
Migration script for AI Platform Trainer codebase.

This script copies and transforms files from the old project structure
to the new enterprise-ready structure, updating imports and ensuring
compatibility.
"""
import re
import shutil
from pathlib import Path


# Mapping from old paths to new paths
FILE_MAPPINGS = {
    # AI components
    "ai_platform_trainer/ai_model/enemy_rl_agent.py": "src/ai_platform_trainer/ml/rl/enemy_agent.py",
    "ai_platform_trainer/ai_model/simple_missile_model.py": "src/ai_platform_trainer/ml/models/missile_model.py",
    "ai_platform_trainer/ai_model/train_missile_model.py": "src/ai_platform_trainer/ml/training/train_missile_model.py",
    "ai_platform_trainer/ai_model/missile_dataset.py": "src/ai_platform_trainer/ml/training/missile_dataset.py",
    "ai_platform_trainer/ai_model/train_enemy_rl.py": "src/ai_platform_trainer/ml/rl/train_enemy_rl.py",
    "ai_platform_trainer/ai_model/training_monitor.py": "src/ai_platform_trainer/ml/training/monitor.py",
    
    # Core components
    "ai_platform_trainer/core/launcher.py": "src/ai_platform_trainer/core/launcher.py",
    "ai_platform_trainer/core/config_manager.py": "src/ai_platform_trainer/core/config.py",
    "ai_platform_trainer/core/data_logger.py": "src/ai_platform_trainer/core/data_logger.py",
    "ai_platform_trainer/core/interfaces.py": "src/ai_platform_trainer/core/interfaces.py",
    
    # Entity components
    "ai_platform_trainer/entities/enemy.py": "src/ai_platform_trainer/entities/enemy.py",
    "ai_platform_trainer/entities/player.py": "src/ai_platform_trainer/entities/player.py",
    "ai_platform_trainer/entities/missile.py": "src/ai_platform_trainer/entities/missile.py",
    "ai_platform_trainer/entities/entity_factory.py": "src/ai_platform_trainer/entities/factory.py",
    
    # Physics components
    "ai_platform_trainer/cpp": "src/ai_platform_trainer/physics/cpp",
    
    # Gameplay components
    "ai_platform_trainer/gameplay/collisions.py": "src/ai_platform_trainer/physics/collisions.py",
    "ai_platform_trainer/gameplay/renderer.py": "src/ai_platform_trainer/rendering/renderer.py",
    "ai_platform_trainer/gameplay/display_manager.py": "src/ai_platform_trainer/rendering/display_manager.py",
    "ai_platform_trainer/gameplay/game.py": "src/ai_platform_trainer/core/game.py",
    
    # Utilities
    "ai_platform_trainer/utils/common_utils.py": "src/ai_platform_trainer/utils/common.py",
    "ai_platform_trainer/utils/helpers.py": "src/ai_platform_trainer/utils/helpers.py",
    "ai_platform_trainer/utils/sprite_manager.py": "src/ai_platform_trainer/rendering/sprite_manager.py",
}

# Import remap patterns
IMPORT_REMAPS = [
    # AI model imports
    (r'from ai_platform_trainer\.ai_model\.([^\s]+) import', r'from ai_platform_trainer.ml.\1 import'),
    (r'import ai_platform_trainer\.ai_model\.([^\s]+)', r'import ai_platform_trainer.ml.\1'),
    
    # Entity imports
    (r'from ai_platform_trainer\.entities\.([^\s]+) import', r'from ai_platform_trainer.entities.\1 import'),
    (r'import ai_platform_trainer\.entities\.([^\s]+)', r'import ai_platform_trainer.entities.\1'),
    
    # Core imports
    (r'from ai_platform_trainer\.core\.([^\s]+) import', r'from ai_platform_trainer.core.\1 import'),
    (r'import ai_platform_trainer\.core\.([^\s]+)', r'import ai_platform_trainer.core.\1'),
    
    # Gameplay remaps to new structure
    (r'from ai_platform_trainer\.gameplay\.renderer import', r'from ai_platform_trainer.rendering.renderer import'),
    (r'from ai_platform_trainer\.gameplay\.display_manager import', r'from ai_platform_trainer.rendering.display_manager import'),
    (r'from ai_platform_trainer\.gameplay\.collisions import', r'from ai_platform_trainer.physics.collisions import'),
    (r'from ai_platform_trainer\.gameplay\.game import', r'from ai_platform_trainer.core.game import'),
    
    # Utils remaps
    (r'from ai_platform_trainer\.utils\.sprite_manager import', r'from ai_platform_trainer.rendering.sprite_manager import'),
]


def ensure_directory(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src_path: str, dst_path: str, update_imports: bool = True) -> None:
    """
    Copy file from src_path to dst_path.
    
    Args:
        src_path: Source file path
        dst_path: Destination file path
        update_imports: Whether to update imports in the file
    """
    src = Path(src_path)
    dst = Path(dst_path)
    
    if not src.exists():
        print(f"Warning: Source file {src} does not exist, skipping")
        return
    
    # Ensure destination directory exists
    ensure_directory(dst.parent)
    
    # Copy file
    print(f"Copying {src} -> {dst}")
    
    if update_imports and src.suffix == '.py':
        # Read source content
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply import remaps
        for pattern, replacement in IMPORT_REMAPS:
            content = re.sub(pattern, replacement, content)
        
        # Write to destination
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        # Simple copy for non-Python files
        shutil.copy2(src, dst)


def copy_directory(src_dir: str, dst_dir: str, update_imports: bool = True) -> None:
    """
    Copy directory from src_dir to dst_dir.
    
    Args:
        src_dir: Source directory path
        dst_dir: Destination directory path
        update_imports: Whether to update imports in Python files
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    
    if not src.exists():
        print(f"Warning: Source directory {src} does not exist, skipping")
        return
    
    # Ensure destination directory exists
    ensure_directory(dst)
    
    print(f"Copying directory {src} -> {dst}")
    
    # Iterate through source directory
    for item in src.glob('**/*'):
        # Calculate relative path
        rel_path = item.relative_to(src)
        target_path = dst / rel_path
        
        if item.is_dir():
            # Create directory
            ensure_directory(target_path)
        else:
            # Copy file
            copy_file(str(item), str(target_path), update_imports=update_imports and item.suffix == '.py')


def migrate_files() -> None:
    """Migrate files according to mappings."""
    for src, dst in FILE_MAPPINGS.items():
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists():
            if src_path.is_dir():
                copy_directory(src, dst)
            else:
                copy_file(src, dst)
        else:
            print(f"Warning: Source {src} does not exist, skipping")


def main() -> None:
    """Main entry point."""
    print("Starting codebase migration...")
    migrate_files()
    print("Migration completed!")


if __name__ == "__main__":
    main()
