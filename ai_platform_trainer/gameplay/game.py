import pygame
import torch
import random
import math
import logging
from typing import Optional, Tuple

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)


class Game:
    """Main class to run the Pixel Pursuit game."""

    def __init__(self) -> None:
        pygame.init()
        self.collision_count = 0  # Tracks number of collisions in play mode
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
        self.data_logger: Optional[DataLogger] = None  # Initialized in training mode

    def run(self) -> None:
        """Main game loop."""
        while self.running:
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update()
                self.renderer.render(
                    self.menu, self.player, self.enemy, self.menu_active
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
                logging.info("Quit event detected. Exiting game.")
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                logging.info("Escape key pressed. Exiting game.")
                self.running = False
            elif self.menu_active:
                selected_action = self.menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)

    def check_menu_selection(self, selected_action: str) -> None:
        """Handle actions selected from the menu."""
        if selected_action == "exit":
            logging.info("Exit action selected from menu.")
            self.running = False
        elif selected_action in ["train", "play"]:
            logging.info(f"'{selected_action}' action selected from menu.")
            self.menu_active = False
            self.start_game(selected_action)

    def start_game(self, mode: str) -> None:
        """
        Initialize game entities and state based on the selected mode (train or play).

        :param mode: "train" or "play"
        """
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

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
        logging.info(f"Randomized speeds with factor {random_speed_factor:.2f}")

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        """
        Initialize player and enemy for play mode.

        :return: Tuple containing PlayerPlay and EnemyPlay instances
        """
        model = SimpleModel(input_size=5, hidden_size=64, output_size=2)
        try:
            model.load_state_dict(
                torch.load(config.MODEL_PATH, map_location=torch.device("cpu"))
            )
            logging.info("Loaded enemy AI model successfully.")
        except Exception as e:
            logging.error(f"Failed to load model from {config.MODEL_PATH}: {e}")
            raise e
        model.eval()
        player = PlayerPlay(self.screen_width, self.screen_height)
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)
        logging.info("Initialized PlayerPlay and EnemyPlay for play mode.")
        return player, enemy

    def _spawn_entities(self) -> None:
        """
        Spawn the player and enemy at random positions ensuring:
        - Both are within the screen margins.
        - They maintain a minimum distance from each other.
        """
        if not self.player or not self.enemy:
            logging.error("Entities not initialized properly. Exiting game.")
            self.running = False
            return

        # Spawn player without any minimum distance requirement
        player_pos = find_valid_spawn_position(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            entity_size=self.player.size,
            margin=config.WALL_MARGIN,
            min_dist=0,  # No minimum distance for player
            other_pos=None,
        )

        # Spawn enemy ensuring minimum distance from player
        enemy_pos = find_valid_spawn_position(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            entity_size=self.enemy.size,
            margin=config.WALL_MARGIN,
            min_dist=config.MIN_DISTANCE,
            other_pos=(self.player.position["x"], self.player.position["y"]),
        )

        self.player.position["x"], self.player.position["y"] = player_pos
        self.enemy.pos["x"], self.enemy.pos["y"] = enemy_pos

        logging.info(f"Spawned player at {player_pos} and enemy at {enemy_pos}.")

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
        collision = player_rect.colliderect(enemy_rect)
        if collision:
            logging.info("Collision detected between player and enemy.")
        return collision

    def play_update(self) -> None:
        """Update logic for play mode."""
        if not self.player.handle_input():
            logging.info("Player requested to quit. Exiting game.")
            self.running = False
            return

        try:
            self.enemy.update_movement(
                self.player.position["x"], self.player.position["y"], self.player.step
            )
            logging.debug("Enemy movement updated in play mode.")
        except Exception as e:
            logging.error(f"Error updating enemy movement: {e}")
            self.running = False
            return

        if self.check_collision():
            logging.info("Collision detected!")
            self.enemy.hide()  # Hide enemy upon collision
            self.respawn_enemy()  # Respawn enemy at new location
            self.collision_count += 1
            logging.info(f"Collision count: {self.collision_count}")

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
            logging.debug("Logged training data point.")

        # Uncomment if you want the game to end upon collision in training mode
        # if collision:
        #     logging.info("Collision detected in training mode! Ending training.")
        #     self.running = False

    def respawn_enemy(self) -> None:
        """
        Respawn the enemy at a new location not too close to the player.
        """
        # Find new position using utility function
        new_pos = find_valid_spawn_position(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            entity_size=self.enemy.size,
            margin=config.WALL_MARGIN,
            min_dist=config.MIN_DISTANCE,
            other_pos=(self.player.position["x"], self.player.position["y"]),
        )

        # Set new position and make enemy visible again
        self.enemy.set_position(new_pos[0], new_pos[1])
        self.enemy.show()

        logging.info(f"Enemy respawned at {new_pos}.")
