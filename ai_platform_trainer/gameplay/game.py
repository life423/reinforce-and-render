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
from ai_platform_trainer.gameplay.utils import (
    compute_normalized_direction,
    find_valid_spawn_position,
)

from typing import Optional, Tuple


class Game:
    """Main class to run the Pixel Pursuit game."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(config.SCREEN_SIZE)
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        # Screen dimensions
        self.screen_width: int = config.SCREEN_WIDTH
        self.screen_height: int = config.SCREEN_HEIGHT

        # UI components
        self.menu = Menu(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.renderer = Renderer(self.screen)

        # Game states
        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None  # "train" or "play"

        # Entities and logger
        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.data_logger: Optional[DataLogger] = None

    def run(self) -> None:
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

        # Save training data if in training mode
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()

    def handle_events(self) -> None:
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
            self.data_logger = DataLogger(config.DATA_PATH)
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            # Randomize speeds for training
            self._randomize_speeds()

            self.player.reset()

            # Spawn entities
            self._spawn_entities()
        else:
            # Play mode: Do not instantiate DataLogger
            self.player, self.enemy = self._init_play_mode()

            self.player.reset()

            # Spawn entities
            self._spawn_entities()

    def _randomize_speeds(self) -> None:
        """Randomize player and enemy speeds for training."""
        random_speed_factor = random.uniform(
            config.RANDOM_SPEED_FACTOR_MIN, config.RANDOM_SPEED_FACTOR_MAX
        )
        self.player.step = int(self.player.step * random_speed_factor)
        self.enemy.base_speed = max(
            config.ENEMY_MIN_SPEED, int(self.enemy.base_speed * random_speed_factor)
        )

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        """
        Initialize player and enemy for play mode.

        :return: Tuple containing PlayerPlay and EnemyPlay instances
        """
        model = SimpleModel(input_size=5, hidden_size=64, output_size=2)
        model.load_state_dict(
            torch.load(config.MODEL_PATH, map_location=torch.device("cpu"))
        )
        model.eval()
        player = PlayerPlay(self.screen_width, self.screen_height)
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)
        return player, enemy

    def _spawn_entities(self) -> None:
        """
        Spawn the player and enemy at random positions ensuring:
        - Both are within the screen margins.
        - They maintain a minimum distance from each other.
        """
        if not self.player or not self.enemy:
            print("Entities not initialized properly.")
            self.running = False
            return

        player_pos = find_valid_spawn_position(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            entity_size=self.player.size,
            margin=config.WALL_MARGIN,
            min_dist=config.MIN_DISTANCE,
            other_pos=None,
        )

        enemy_pos = find_valid_spawn_position(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            entity_size=self.enemy.size,
            margin=config.WALL_MARGIN,
            min_dist=config.MIN_DISTANCE,
            other_pos=player_pos,
        )

        self.player.position["x"], self.player.position["y"] = player_pos
        self.enemy.pos["x"], self.enemy.pos["y"] = enemy_pos

    def update(self) -> None:
        """Update game state depending on the mode."""
        if self.mode == "train":
            self.training_update()
        elif self.mode == "play":
            self.play_update()

    def check_collision(self) -> bool:
        """
        Check if the player and enemy collide.

        :return: True if collision occurs, False otherwise.
        """
        if not self.player or not self.enemy:
            return False

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

    def play_update(self) -> None:
        """Update logic for play mode."""
        if not self.player.handle_input():
            self.running = False
            return

        try:
            self.enemy.update_movement(
                self.player.position["x"], self.player.position["y"], self.player.step
            )
        except Exception as e:
            print(f"Error updating enemy movement: {e}")
            self.running = False
            return

        if self.check_collision():
            print("Collision detected!")
            self.running = False

    def training_update(self) -> None:
        """Update logic for training mode."""
        self.player.update(self.enemy.pos["x"], self.enemy.pos["y"])

        px = self.player.position["x"]
        py = self.player.position["y"]
        ex = self.enemy.pos["x"]
        ey = self.enemy.pos["y"]

        # Compute direction and move enemy toward player
        action_dx, action_dy = compute_normalized_direction(px, py, ex, ey)
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
                    "dist": math.hypot(
                        px - self.enemy.pos["x"], py - self.enemy.pos["y"]
                    ),
                }
            )

        # if collision:
        #     print("Collision detected in training mode!")
        #     self.running = False
