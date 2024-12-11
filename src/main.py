from gameplay.game import Game
import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set Pygame to use a dummy video driver if running in an environment without a display
# os.environ["SDL_VIDEODRIlsER"] = "dummy"

if __name__ == "__main__":
    game = Game()
    game.run()  # No arguments needed

