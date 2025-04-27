from ai_platform_trainer.engine.display_manager import DisplayManager
from ai_platform_trainer.engine.input_handler import InputHandler
from ai_platform_trainer.engine.renderer import Renderer
from ai_platform_trainer.engine.entities.entity_factory import EntityFactory

class Game:
    def __init__(self):
        self.display = DisplayManager(800, 600)
        self.input = InputHandler()
        self.renderer = Renderer(self.display.get_screen())
        self.is_running = True
        self.player = EntityFactory.create_player()
        self.enemies = EntityFactory.create_enemies(3)

    def run(self) -> None:
        while self.is_running:
            actions = self.input.get_actions()
            if actions.get('quit'):
                self.is_running = False
                continue

            # 1) Update
            self.player.update(actions)
            for enemy in self.enemies:
                enemy.update()

            # 2) Render
            self.renderer.clear((0, 0, 0))
            self.renderer.draw(self.player)
            for enemy in self.enemies:
                self.renderer.draw(enemy)
            self.renderer.present()

            self.display.tick(60)

        self.display.quit()
