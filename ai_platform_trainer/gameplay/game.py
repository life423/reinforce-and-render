import pygame
import torch
import random
import math

from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import Player as PlayerTraining
from ai_platform_trainer.entities.enemy_play import Enemy as EnemyPlay
from ai_platform_trainer.entities.enemy_training import Enemy as EnemyTrain
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.ai_model.model_definition.model import SimpleModel


class Game:
    """Main class to run the Pixel Pursuit game."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        )
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        self.screen_width = config.SCREEN_WIDTH
        self.screen_height = config.SCREEN_HEIGHT

        self.menu = Menu(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.renderer = Renderer(self.screen)
        # Removed DataLogger initialization from here

        # Game states
        self.running = True
        self.menu_active = True
        self.mode = None  # "train" or "play"

        # Entities will be initialized in start_game
        self.player = None
        self.enemy = None
        self.data_logger = None  # Initialize as None

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
            self.clock.tick(config.FRAME_RATE)

        # Only save if in training mode
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()

    def handle_events(self):
        """Handle all window and menu-related events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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
        """
        Initialize game entities and state based on the selected mode (train or play).

        :param mode: "train" or "play"
        """
        self.mode = mode
        print(f"Starting game in '{mode}' mode.")

        if mode == "train":
            # Instantiate DataLogger only in training mode
            self.data_logger = DataLogger("data/raw/training_data.json")
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            # Randomize speeds for training
            random_speed_factor = random.uniform(0.8, 1.2)
            self.player.step = int(self.player.step * random_speed_factor)
            self.enemy.base_speed = max(
                2, int(self.enemy.base_speed * random_speed_factor)
            )
        else:
            # Play mode: Do not instantiate DataLogger
            model = SimpleModel(input_size=5, hidden_size=64, output_size=2)
            model.load_state_dict(
                torch.load(
                    "models/enemy_ai_model.pth", map_location=torch.device("cpu")
                )
            )
            model.eval()
            self.player = PlayerPlay(self.screen_width, self.screen_height)
            self.enemy = EnemyPlay(self.screen_width, self.screen_height, model)

        self.player.reset()

        # Define margins and min distance
        wall_margin = 50
        min_dist = 100

        player_min_x = wall_margin
        player_max_x = self.screen_width - self.player.size - wall_margin
        player_min_y = wall_margin
        player_max_y = self.screen_height - self.player.size - wall_margin

        enemy_min_x = wall_margin
        enemy_max_x = self.screen_width - self.enemy.size - wall_margin
        enemy_min_y = wall_margin
        enemy_max_y = self.screen_height - self.enemy.size - wall_margin

        placed = False
        while not placed:
            px = random.randint(player_min_x, player_max_x)
            py = random.randint(player_min_y, player_max_y)
            ex = random.randint(enemy_min_x, enemy_max_x)
            ey = random.randint(enemy_min_y, enemy_max_y)

            dist = math.sqrt((px - ex) ** 2 + (py - ey) ** 2)
            if dist >= min_dist:
                self.player.position["x"] = px
                self.player.position["y"] = py
                self.enemy.pos["x"] = ex
                self.enemy.pos["y"] = ey
                placed = True

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
            self.enemy.pos["x"],
            self.enemy.pos["y"],
            self.enemy.size,
            self.enemy.size,
        )
        return player_rect.colliderect(enemy_rect)

    def play_update(self):
        """Update logic for play mode."""
        if not self.player.handle_input():
            self.running = False
            return

        self.enemy.update_movement(
            self.player.position["x"], self.player.position["y"], self.player.step
        )
        if self.check_collision():
            print("Collision detected!")
            self.running = False

    def training_update(self):
        """Update logic for training mode."""
        self.player.update(self.enemy.pos["x"], self.enemy.pos["y"])

        px = self.player.position["x"]
        py = self.player.position["y"]
        ex = self.enemy.pos["x"]
        ey = self.enemy.pos["y"]

        direction_x = px - ex
        direction_y = py - ey
        dist = math.sqrt(direction_x**2 + direction_y**2)

        if dist > 0:
            action_dx = direction_x / dist
            action_dy = direction_y / dist
        else:
            action_dx = 0
            action_dy = 0

        speed = self.enemy.base_speed
        self.enemy.pos["x"] += action_dx * speed
        self.enemy.pos["y"] += action_dy * speed

        collision = self.check_collision()

        # Log training data
        if self.data_logger:
            self.data_logger.log(
                {
                    "mode": "train",
                    "player_x": px,
                    "player_y": py,
                    "enemy_x": self.enemy.pos["x"],
                    "enemy_y": self.enemy.pos["y"],
                    "action_dx": action_dx,
                    "action_dy": action_dy,
                    "collision": collision,
                    "dist": math.sqrt(
                        (px - self.enemy.pos["x"]) ** 2
                        + (py - self.enemy.pos["y"]) ** 2
                    ),
                }
            )

        if collision:
            print("Collision detected in training mode!")
            self.running = False


if __name__ == "__main__":
    game = Game()
    game.run()
