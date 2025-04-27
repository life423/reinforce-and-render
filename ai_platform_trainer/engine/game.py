# ai_platform_trainer/engine/game.py

from ai_platform_trainer.engine.display_manager import DisplayManager
from ai_platform_trainer.engine.input_handler import InputHandler
from ai_platform_trainer.engine.renderer import Renderer
from ai_platform_trainer.engine.entities.entity import Entity

class Game:
    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.display = DisplayManager(width, height)
        self.input = InputHandler()
        self.renderer = Renderer()
        self.is_running = True

        # ─── TEST ENTITIES ─────────────────────────────────────
        # Spawn a red circle and a green circle to prove rendering works:
        self.entities = [
            Entity(position=(100, 100), color=(255,  50,  50), radius=20),
            Entity(position=(300, 300), color=( 50, 255,  50), radius=30),
        ]
        # ─────────────────────────────────────────────────────────

    def run(self) -> None:
        while self.is_running:
            actions = self.input.get_actions()
            if actions.get("quit"):
                self.is_running = False

            # Clear screen to black
            self.display.get_screen().fill((0, 0, 0))

            # ─── RENDER ENTITIES ────────────────────────────────────
            self.renderer.render(self.display.get_screen(), self.entities)
            # ─────────────────────────────────────────────────────────

            self.display.update()
            self.display.tick(60)

        self.display.quit()
