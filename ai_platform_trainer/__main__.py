from ai_platform_trainer.engine.menu import MainMenu
from ai_platform_trainer.engine.game import Game
import subprocess
import sys
import pygame
import os

def _spawn(module: str, headless=False):
    """
    Spawn a subprocess for training or demo modules.
    
    Args:
        module: The module to run
        headless: Whether to run in headless mode
    """
    cmd = [sys.executable, "-m", module]
    if headless:
        cmd.append("--headless")
    subprocess.Popen(
        cmd, 
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    )

def main():
    """Main application entry point."""
    # Initialize pygame
    pygame.init()
    
    # Set fullscreen by default with a fallback resolution
    # Using (0, 0) can cause issues on some systems, so we provide a fallback size
    flags = pygame.FULLSCREEN | pygame.SCALED | pygame.DOUBLEBUF
    try:
        # First try with native resolution
        info = pygame.display.Info()
        if hasattr(info, 'current_w') and hasattr(info, 'current_h'):
            # Use current display dimensions if available
            screen = pygame.display.set_mode(
                (info.current_w, info.current_h), 
                flags, 
                vsync=1
            )
        else:
            # Fall back to a common fullscreen resolution
            screen = pygame.display.set_mode((1920, 1080), flags, vsync=1)
    except pygame.error:
        # If fullscreen fails, fall back to windowed mode
        flags = pygame.SCALED | pygame.DOUBLEBUF
        screen = pygame.display.set_mode((800, 600), flags, vsync=1)
    
    # Set window title
    pygame.display.set_caption("Reinforce-and-Render")
    
    # Main application loop
    while True:
        # Create and run menu
        menu = MainMenu(screen)
        choice = menu.run()
        
        # Exit if no choice was made (Quit or window X)
        if not choice:
            break

        # Handle menu choices
        match choice:
            case "Play RL Game":
                Game().run()
            case "Train RL Model (live)":
                _spawn("ai_platform_trainer.agents.training_loop")
            case "Train RL Model (headless)":
                _spawn("ai_platform_trainer.agents.training_loop", headless=True)
            case "Train Supervised Model (live)":
                _spawn("ai_platform_trainer.supervised.supervised_demo")
            case "Train Supervised Model (headless)":
                _spawn("ai_platform_trainer.supervised.supervised_demo", headless=True)
        # Loop back to menu when child process exits or Game ends

    # Clean up pygame
    pygame.quit()

if __name__ == "__main__":
    main()
