"""
Main entry point for AI Platform Trainer.

This module initializes and runs the game using the unified launcher.
The unified launcher provides support for multiple game initialization methods
and handles fallbacks if a specific method fails.
"""
from ai_platform_trainer.engine.core.unified_launcher import main as _unified_main


# Define main function here to ensure it's properly exposed at the module level
def main():
    """
    Main entry point for the game.
    
    This function is called by the console script entry point defined in setup.py.
    It delegates to the unified launcher's main function.
    """
    _unified_main()


if __name__ == "__main__":
    main()
