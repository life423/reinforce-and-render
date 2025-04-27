from ai_platform_trainer.engine.menu import MainMenu
from ai_platform_trainer.engine.game import Game
import subprocess, sys, pygame, os

def _spawn(module: str, headless=False):
    cmd = [sys.executable, "-m", module]
    if headless:
        cmd.append("--headless")
    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE if os.name=="nt" else 0)

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.SCALED | pygame.DOUBLEBUF, vsync=1)

    while True:
        choice = MainMenu(screen).run()
        if not choice:   # Quit or window X
            break

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
        # loop back to menu when child process exits or Game ends

    pygame.quit()

if __name__ == "__main__":
    main()
