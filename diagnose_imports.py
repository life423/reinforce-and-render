"""
Diagnostic script to check import paths and module availability
"""
import sys
import os

print("Python version:", sys.version)
print("Executable:", sys.executable)
print("\nSystem Path:")
for p in sys.path:
    print("  -", p)

print("\nTrying imports...")
try:
    import ai_platform_trainer
    print("✓ Successfully imported ai_platform_trainer")
except ImportError as e:
    print("✗ Failed to import ai_platform_trainer:", e)

try:
    from ai_platform_trainer.core import config_manager as core_config
    print("✓ Successfully imported ai_platform_trainer.core.config_manager")
except ImportError as e:
    print("✗ Failed to import ai_platform_trainer.core.config_manager:", e)

try:
    from ai_platform_trainer.engine.core import config_manager as engine_config
    print("✓ Successfully imported ai_platform_trainer.engine.core.config_manager")
except ImportError as e:
    print("✗ Failed to import ai_platform_trainer.engine.core.config_manager:", e)

try:
    import config_manager as root_config
    print("✓ Successfully imported root config_manager")
except ImportError as e:
    print("✗ Failed to import root config_manager:", e)

print("\nChecking package directory structure:")
ai_platform_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_platform_trainer')
if os.path.exists(ai_platform_dir):
    print(f"✓ ai_platform_trainer directory exists at {ai_platform_dir}")
    # Check for __init__.py files
    missing_inits = []
    for root, dirs, files in os.walk(ai_platform_dir):
        if "__init__.py" not in files and "__pycache__" not in root:
            rel_path = os.path.relpath(root, os.path.dirname(ai_platform_dir))
            missing_inits.append(rel_path)
    
    if missing_inits:
        print("✗ Missing __init__.py files in these directories:")
        for path in missing_inits:
            print(f"  - {path}")
    else:
        print("✓ All subdirectories have __init__.py files")
else:
    print(f"✗ ai_platform_trainer directory not found at {ai_platform_dir}")