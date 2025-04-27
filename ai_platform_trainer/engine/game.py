from ai_platform_trainer.engine.display_manager import DisplayManager
from ai_platform_trainer.engine.input_handler import InputHandler
from ai_platform_trainer.engine.renderer import Renderer
from ai_platform_trainer.engine.entities.entity_factory import EntityFactory
from ai_platform_trainer.engine.physics import PhysicsSystem

class Game:
    def __init__(self):
        """Initialize the game with display, input handling, physics and entities."""
        self.width = 800
        self.height = 600
        
        # Initialize systems
        self.display = DisplayManager(self.width, self.height)
        self.input = InputHandler()
        self.renderer = Renderer(self.display.get_screen())
        self.physics = PhysicsSystem(self.width, self.height)
        self.is_running = True
        
        # Create entities with physics bodies
        self.player = EntityFactory.create_player(physics=self.physics)
        self.enemies = EntityFactory.create_enemies(3, physics=self.physics)

    def run(self) -> None:
        """Run the main game loop."""
        # Access the clock directly
        clock = self.display.clock
        
        while self.is_running:
            # Calculate delta time in seconds
            dt = min(clock.get_time() / 1000.0, 0.1)  # Cap dt at 0.1 seconds
            
            # Process input
            actions = self.input.get_actions()
            if actions.get('quit'):
                self.is_running = False
                continue

            # 1) Update physics and entities
            self.physics.step_space(dt)  # Step physics simulation
            
            # Update entities (applies forces based on input for player)
            self.player.update(actions)
            for enemy in self.enemies:
                enemy.update()

            # 2) Render
            self.renderer.clear((0, 0, 0))
            self.renderer.draw(self.player)
            for enemy in self.enemies:
                self.renderer.draw(enemy)
            self.renderer.present()

            # Cap at 60 FPS
            self.display.tick(60)

        self.display.quit()
