"""
Launcher Module (Adapter)

This adapter module provides backward compatibility with the three launcher patterns:
1. Standard Launcher
2. Dependency Injection Launcher
3. State Machine Launcher

It forwards to the canonical implementation in engine/core/unified_launcher.py.

DEPRECATED: Use ai_platform_trainer.engine.core.unified_launcher instead.
"""

import os
import warnings

from ai_platform_trainer.engine.core.unified_launcher import main
from ai_platform_trainer.engine.core.launcher_di import register_services

# Determine which launcher mode to use based on the module name
module_name = __name__.split('.')[-1]

if module_name == 'launcher_di' or os.environ.get('AI_PLATFORM_USE_DI', 'False') == 'True':
    # Dependency Injection launcher mode
    warnings.warn(
        "Importing from ai_platform_trainer.core.launcher_di is deprecated. "
        "Use ai_platform_trainer.engine.core.unified_launcher instead.",
        DeprecationWarning,
        stacklevel=2
    )
    os.environ["AI_PLATFORM_LAUNCHER_MODE"] = "DI"
    __all__ = ["main", "register_services"]
    
elif module_name == 'launcher_refactored' or os.environ.get('AI_PLATFORM_USE_STATE_MACHINE', 'False') == 'True':
    # State Machine launcher mode
    warnings.warn(
        "Importing from ai_platform_trainer.core.launcher_refactored is deprecated. "
        "Use ai_platform_trainer.engine.core.unified_launcher instead.",
        DeprecationWarning,
        stacklevel=2
    )
    os.environ["AI_PLATFORM_LAUNCHER_MODE"] = "STATE_MACHINE"
    __all__ = ["main"]
    
else:
    # Standard launcher mode (default)
    warnings.warn(
        "Importing from ai_platform_trainer.core.launcher is deprecated. "
        "Use ai_platform_trainer.engine.core.unified_launcher instead.",
        DeprecationWarning,
        stacklevel=2
    )
    __all__ = ["main"]
