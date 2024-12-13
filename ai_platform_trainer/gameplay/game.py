import pygame
import torch
import random
import math
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import Player as PlayerTraining
from ai_platform_trainer.entities.enemy_play import Enemy as EnemyPlay
from ai_platform_trainer.entities.enemy_training import Enemy as EnemyTrain
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.ai_model.model_definition.model import SimpleModel

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WINDOW_TITLE = "Pixel Pursuit"
FRAME_RATE = 60
DATA_PATH = "data/training_data.json"


class Game:
    """Main class to run the Pixel Pursuit game."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        self.menu = Menu(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.renderer = Renderer(self.screen)
        self.data_logger = DataLogger(DATA_PATH)

        # Game states
        self.running = True
        self.menu_active = True
        self.mode = None  # "train" or "play"

    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update()
                self.renderer.render(
                    self.menu, self.player, self.enemy, self.menu_active, self.screen
                )

            pygame.display.flip()
            self.clock.tick(FRAME_RATE)

        # Only save if in training mode
        if self.mode == "train":
            self.data_logger.save()
        # if escape is pressed, the game will close

        pygame.quit()

    def handle_events(self):
        """Handle all window and menu-related events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # Pressing ESC will now stop the game universally
                self.running = False
            elif self.menu_active:
                selected_action = self.menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)

    def check_menu_selection(self, selected_action: str) -> None:
        """Handle actions selected from the menu."""
        if selected_action == "exit":
            self.running = False
        elif selected_action in ["train", "play"]:
            self.menu_active = False
            self.start_game(selected_action)

    def start_game(self, mode: str) -> None:
        self.mode = mode
        print(mode)

        if mode == "train":
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            # Increase complexity: Randomize speeds each run in training mode
            random_speed_factor = random.uniform(0.8, 1.2)  # +/-20% speed
            self.player.step = int(self.player.step * random_speed_factor)
            self.enemy.base_speed = max(
                2, int((self.enemy.base_speed) * random_speed_factor)
            )

            # Randomize starting positions with constraints
            # (Add your logic for random spawn with margin and min distance)

        else:
            model = SimpleModel(
                input_size=5, hidden_size=64, output_size=2
            )  # If you're using distance
            model.load_state_dict(
                torch.load("models/enemy_ai_model.pth", weights_only=True)
            )
            model.eval()
            self.player = PlayerPlay(self.screen_width, self.screen_height)
            self.enemy = EnemyPlay(self.screen_width, self.screen_height, model)

        self.player.reset()

        # Define margins and min distance
        wall_margin = 50
        min_dist = 100  # Minimum distance between player and enemy

        # Ensure both entities have room (their size + margin)
        # Player:
        player_min_x = wall_margin
        player_max_x = self.screen_width - self.player.size - wall_margin
        player_min_y = wall_margin
        player_max_y = self.screen_height - self.player.size - wall_margin

        # Enemy:
        enemy_min_x = wall_margin
        enemy_max_x = self.screen_width - self.enemy.size - wall_margin
        enemy_min_y = wall_margin
        enemy_max_y = self.screen_height - self.enemy.size - wall_margin

        # Pick random positions until constraints are met
        # This loop ensures we eventually find suitable positions
        placed = False
        while not placed:
            # Random player position within margins
            px = random.randint(player_min_x, player_max_x)
            py = random.randint(player_min_y, player_max_y)

            # Random enemy position within margins
            ex = random.randint(enemy_min_x, enemy_max_x)
            ey = random.randint(enemy_min_y, enemy_max_y)

            # Compute distance between them
            dist = math.sqrt((px - ex) ** 2 + (py - ey) ** 2)

            if dist >= min_dist:
                # Suitable positions found
                self.player.position["x"] = px
                self.player.position["y"] = py
                self.enemy.pos["x"] = ex
                self.enemy.pos["y"] = ey
                placed = True

        # Now we have player and enemy spawned within margins and not too close to each other.

    def update(self):
        """Update game state depending on the mode."""
        if self.mode == "train":
            self.training_update()
        elif self.mode == "play":
            self.play_update()

    def check_collision(self) -> bool:
        """Check if the player and enemy collide."""
        player_rect = pygame.Rect(
            self.player.position["x"],
            self.player.position["y"],
            self.player.size,
            self.player.size,
        )
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"], self.enemy.pos["y"], self.enemy.size, self.enemy.size
        )
        return player_rect.colliderect(enemy_rect)

    def play_update(self):
        """Update logic for play mode."""
        # Let the player handle input directly
        if not self.player.handle_input():
            self.running = False
            return

        # Enemy uses the model to pick actions in EnemyPlay.update_movement()
        self.enemy.update_movement(
            self.player.position["x"], self.player.position["y"], self.player.step
        )
        if self.check_collision():
            print("Collision detected!")
            self.running = False

    def training_update(self):
        """Update logic for training mode."""
        self.player.update(self.enemy.pos["x"], self.enemy.pos["y"])
        old_enemy_x = self.enemy.pos["x"]
        old_enemy_y = self.enemy.pos["y"]
        self.enemy.update_movement(
            self.player.position["x"], self.player.position["y"], self.player.step
        )

        action_dx = self.enemy.pos["x"] - old_enemy_x
        action_dy = self.enemy.pos["y"] - old_enemy_y
        collision = self.check_collision()

        self.data_logger.log(
            {
                "mode": "train",
                "player_x": self.player.position["x"],
                "player_y": self.player.position["y"],
                "enemy_x": self.enemy.pos["x"],
                "enemy_y": self.enemy.pos["y"],
                "action_dx": action_dx,
                "action_dy": action_dy,
                "collision": collision,
                "dist": math.sqrt(
                    (self.player.position["x"] - self.enemy.pos["x"]) ** 2
                    + (self.player.position["y"] - self.enemy.pos["y"]) ** 2
                ),
            }
        )


if __name__ == "__main__":
    game = Game()
    game.run()
