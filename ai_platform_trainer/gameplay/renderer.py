"""
Compatibility module for AI Platform Trainer renderer.

This module re-exports the Renderer class from the new location to maintain
backward compatibility with existing code during the refactoring process.
"""

# Import from new location and re-export directly
from ai_platform_trainer.engine.rendering.renderer import Renderer

# Explicitly define what this module exports
__all__ = ['Renderer']
