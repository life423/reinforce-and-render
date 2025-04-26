"""
Main entry point for AI Platform Trainer.

This module initializes and runs the game using the unified launcher.
The unified launcher provides support for multiple game initialization methods
and handles fallbacks if a specific method fails.
"""
import sys


# Define main function here to ensure it's properly exposed at the module level
def main():
    """
    Main entry point for the game.
    
    This function is called by the console script entry point defined in setup.py.
    It delegates to the unified launcher's main function.
    
    Handles potential import errors and provides helpful error messages
    to assist users in resolving dependency issues.
    """
    try:
        # Try importing pygame to check if it's available
        try:
            import pygame
            pygame_version = pygame.version.ver
            print(f"Using pygame version {pygame_version}")
        except ImportError as e:
            print("=" * 80)
            print("ERROR: Pygame is not properly installed or configured.")
            print(f"Error details: {e}")
            print("\nTo fix this issue, try reinstalling pygame:")
            print("  pip uninstall pygame")
            print("  pip install pygame>=2.5.0")
            print("\nIf you're using a virtual environment, make sure it's activated.")
            print("=" * 80)
            sys.exit(1)
            
        # If pygame is available, import and run the unified launcher
        from ai_platform_trainer.engine.core.unified_launcher import main as _unified_main
        _unified_main()
    except Exception as e:
        print("=" * 80)
        print(f"ERROR: Failed to start the game: {e}")
        print("\nIf this error persists, please check the following:")
        print("1. All dependencies are installed (see requirements.txt)")
        print("2. Your Python environment is properly configured")
        print("3. The game files haven't been corrupted")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
