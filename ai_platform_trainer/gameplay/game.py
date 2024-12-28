import math
import random
import os
from typing import Optional, Tuple

import pygame
import torch
import logging

from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.ai_model.model_definition.simple_model import SimpleModel

from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.spawner import (
    spawn_entities,
    respawn_enemy_with_fade_in,
)
from ai_platform_trainer.gameplay.utils import (
    compute_normalized_direction,
    find_valid_spawn_position,
)

from ai_platform_trainer.gameplay.modes.training_mode import TrainingModeManager
from ai_platform_trainer.ai_model.train_missile_model import SimpleMissileModel


class Game:
    """Main class to run the Pixel Pursuit game."""

    def __init__(self) -> None:
        # -------------------------
        # (A) Basic Game Setup
        # -------------------------
        pygame.init()
        self.collision_count = 0
        self.screen = pygame.display.set_mode(config.SCREEN_SIZE)
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        self.screen_width: int = config.SCREEN_WIDTH
        self.screen_height: int = config.SCREEN_HEIGHT

        self.menu = Menu(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        self.renderer = Renderer(self.screen)

        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None

        # -------------------------
        # (B) Key Game Entities
        # -------------------------
        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.data_logger: Optional[DataLogger] = None

        # Store the missile AI model here if loaded
        self.missile_model: Optional[SimpleMissileModel] = None

        # -------------------------
        # (C) Respawn & Timing
        # -------------------------
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        self.last_shot_time = 0
        self.shot_cooldown = 500

    def run(self) -> None:
        """Main game loop."""
        while self.running:
            current_time = pygame.time.get_ticks()
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update(current_time)
                self.renderer.render(
                    self.menu, self.player, self.enemy, self.menu_active
                )

            pygame.display.flip()
            self.clock.tick(config.FRAME_RATE)

        # Save any training data if we're in training mode
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()

    def start_game(self, mode: str) -> None:
        """
        Starts the game in the specified mode (train or play).
        If the missile model file exists, we load it; otherwise, we skip it.
        """
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

        # -------------------------
        # (A) TRAIN MODE
        # -------------------------
        if mode == "train":
            self.data_logger = DataLogger(config.DATA_PATH)
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            # Flag indicating we're capturing missile data for training
            self.train_missile = True

            self._apply_speed_variation()
            self.player.reset()
            spawn_entities(self)

            # Create a TrainingModeManager for logging, etc.
            self.training_mode_manager = TrainingModeManager(self)

        # -------------------------
        # (B) PLAY MODE
        # -------------------------
        else:
            self.player, self.enemy = self._init_play_mode()
            self.player.reset()
            spawn_entities(self)

            # Check if there's a missile model we can use
            missile_model_path = "models/missile_model.pth"
            if os.path.isfile(missile_model_path):
                logging.info(
                    f"Found missile model at '{missile_model_path}'. Loading it..."
                )
                try:
                    self.missile_model = SimpleMissileModel()
                    self.missile_model.load_state_dict(
                        torch.load(missile_model_path, map_location="cpu")
                    )
                    self.missile_model.eval()
                    logging.info("Missile model loaded successfully.")
                except Exception as e:
                    logging.error(f"Failed to load missile model: {e}")
                    self.missile_model = None
            else:
                logging.warning(
                    f"No missile model found at '{missile_model_path}'. Skipping missile AI."
                )

    def _apply_speed_variation(self) -> None:
        """Apply speed variation for player and enemy in training mode."""
        random_speed_factor = random.uniform(
            config.RANDOM_SPEED_FACTOR_MIN, config.RANDOM_SPEED_FACTOR_MAX
        )
        self.player.step = int(self.player.step * random_speed_factor)
        self.enemy.base_speed = max(
            config.ENEMY_MIN_SPEED, int(self.enemy.base_speed * random_speed_factor)
        )
        logging.info(f"Applied speed variation with factor {random_speed_factor:.2f}")

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
                # Menu is active, handle menu input
                selected_action = self.menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)
            elif event.type == pygame.KEYDOWN:
                # If user presses space, shoot a missile
                if event.key == pygame.K_SPACE and self.player:
                    self.player.shoot_missile()

    def check_menu_selection(self, selected_action: str) -> None:
        """Handle actions selected from the menu."""
        if selected_action == "exit":
            logging.info("Exit action selected from menu.")
            self.running = False
        elif selected_action in ["train", "play"]:
            logging.info(f"'{selected_action}' action selected from menu.")
            self.menu_active = False
            self.start_game(selected_action)

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        """
        Initialize player and enemy for play mode.
        Loads the enemy AI model from config.MODEL_PATH if available.
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

    def update(self, current_time: int) -> None:
        """Update game state depending on the mode."""
        if self.mode == "train":
            # In training mode, the TrainingModeManager does missile + enemy logic
            self.training_mode_manager.update()
        elif self.mode == "play":
            # In play mode, we do our usual game updates
            self.play_update(current_time)
            self.handle_respawn(current_time)

            # If the enemy is fading in, keep updating fade
            if self.enemy and self.enemy.fading_in:
                self.enemy.update_fade_in(current_time)

            # Update missiles and check collisions each frame
            if self.player:
                self.player.update_missiles()
            self.check_missile_collisions()

    def check_collision(self) -> bool:
        """Check if the player and enemy collide."""
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

    def play_update(self, current_time: int) -> None:
        """Update logic for play mode."""
        # -------------------------
        # (A) Player Input
        # -------------------------
        if self.player and not self.player.handle_input():
            logging.info("Player requested to quit. Exiting game.")
            self.running = False
            return

        # -------------------------
        # (B) Enemy Movement
        # -------------------------
        if self.enemy:
            try:
                self.enemy.update_movement(
                    self.player.position["x"] if self.player else 0,
                    self.player.position["y"] if self.player else 0,
                    self.player.step if self.player else 0,
                    current_time,
                )
                logging.debug("Enemy movement updated in play mode.")
            except Exception as e:
                logging.error(f"Error updating enemy movement: {e}")
                self.running = False
                return

        # -------------------------
        # (C) Check Player-Enemy Collision
        # -------------------------
        if self.check_collision():
            logging.info("Collision detected!")
            if self.enemy:
                self.enemy.hide()
            self.is_respawning = True
            self.respawn_timer = current_time + self.respawn_delay
            logging.info(f"Enemy will respawn in {self.respawn_delay} ms.")

        # -------------------------
        # (D) Apply Missile AI (Optional)
        # -------------------------
        if self.missile_model and self.player and self.player.missiles:
            # Loop over all active missiles
            for missile in self.player.missiles:
                # Suppose you want to feed [player_x, player_y, enemy_x, enemy_y, missile_x, missile_y, angle]
                # If you track the missile's angle, you'd need to store it in missile or compute from vx, vy.
                current_angle = math.atan2(missile.vy, missile.vx)

                input_state = torch.tensor(
                    [
                        [
                            self.player.position["x"],  # player_x
                            self.player.position["y"],  # player_y
                            self.enemy.pos["x"],  # enemy_x
                            self.enemy.pos["y"],  # enemy_y
                            missile.pos["x"],  # missile_x
                            missile.pos["y"],  # missile_y
                            current_angle,  # missile_angle
                        ]
                    ],
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    predicted_action = self.missile_model(input_state)
                angle_delta = predicted_action.item()

                # Update missile's angle
                new_angle = current_angle + angle_delta

                # Example: fix speed or let speed vary
                speed = 5.0
                missile.vx = math.cos(new_angle) * speed
                missile.vy = math.sin(new_angle) * speed

    def handle_respawn(self, current_time: int) -> None:
        """
        Handle the respawn of the enemy after a delay.
        """
        if (
            self.is_respawning
            and current_time >= self.respawn_timer
            and self.enemy
            and self.player
        ):
            respawn_enemy_with_fade_in(self, current_time)

    def check_missile_collisions(self) -> None:
        """
        Check if a missile collides with the enemy, and trigger respawn if needed.
        """
        if not self.enemy or not self.player:
            return

        def respawn_callback():
            self.is_respawning = True
            self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
            logging.info(
                f"Enemy will respawn in {self.respawn_delay} ms due to missile hit."
            )

        handle_missile_collisions(self.player, self.enemy, respawn_callback)
