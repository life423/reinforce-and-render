"""
Configuration Management Module (Root-level Adapter)

This is an adapter module that forwards to the canonical implementation
in ai_platform_trainer.engine.core.config_manager for backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.config_manager instead.
"""
import warnings

# Import the canonical module
import ai_platform_trainer.engine.core.config_manager as core_config_manager

# Re-export all public attributes
__all__ = [name for name in dir(core_config_manager) if not name.startswith('_')]
globals().update({name: getattr(core_config_manager, name) for name in __all__})

# Add deprecation warning
warnings.warn(
    "Importing from root-level config_manager is deprecated. "
    "Use ai_platform_trainer.engine.core.config_manager instead.",
    DeprecationWarning,
    stacklevel=2
)