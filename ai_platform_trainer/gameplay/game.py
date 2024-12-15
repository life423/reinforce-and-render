import math
import random
from typing import Optional, Tuple

import pygame
import torch
import logging

from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.ai_model.model_definition.model import SimpleModel
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import spawn_entities
from ai_platform_trainer.gameplay.utils import (
    compute_normalized_direction,
    find_valid_spawn_position,
)


class Game:
    """Main class to run the Pixel Pursuit game."""

    def __init__(self) -> None:
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

        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.data_logger: Optional[DataLogger] = None

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

        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()

    def start_game(self, mode: str) -> None:
        """Start the game in the given mode ('train' or 'play')."""
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

        if mode == "train":
            self.data_logger = DataLogger(config.DATA_PATH)
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            self.train_missile = True

            self._apply_speed_variation()
            self.player.reset()
            spawn_entities(self)
        else:
            self.player, self.enemy = self._init_play_mode()
            self.player.reset()
            spawn_entities(self)

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
                selected_action = self.menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.player:
                    self.player.shoot_missile()

    def handle_shoot(self) -> None:
        """
        Handle shooting a missile towards the current mouse position.
        (If you intend to use this function, call it from where needed.)
        """
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time >= self.shot_cooldown and self.player:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            self.player.shoot_missile(mouse_x, mouse_y)
            self.last_shot_time = current_time
            logging.info(f"Player shot missile towards ({mouse_x}, {mouse_y}).")

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

    def update(self, current_time: int) -> None:
        """Update game state depending on the mode."""
        if self.mode == "train":
            self.training_update()
        elif self.mode == "play":
            self.play_update(current_time)
            # Removed handle_respawn call as spawning is now handled externally
            if self.enemy and self.enemy.fading_in:
                self.enemy.update_fade_in(current_time)
            if self.enemy:
                enemy_pos = (
                    (self.enemy.pos["x"], self.enemy.pos["y"])
                    if self.enemy.visible
                    else (0, 0)
                )
                if self.player:
                    self.player.update_missiles(enemy_pos)
            self.check_missile_collisions()

    def check_collision(self) -> bool:
        """
        Check if the player and enemy collide.
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

    def play_update(self, current_time: int) -> None:
        """Update logic for play mode."""
        if self.player and not self.player.handle_input():
            logging.info("Player requested to quit. Exiting game.")
            self.running = False
            return

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

        if self.check_collision():
            logging.info("Collision detected!")
            if self.enemy:
                self.enemy.hide()
            self.is_respawning = True
            self.respawn_timer = current_time + self.respawn_delay
            logging.info(f"Enemy will respawn in {self.respawn_delay} ms.")

    def training_update(self):
        if self.enemy and self.player:
            self.player.update(self.enemy.pos["x"], self.enemy.pos["y"])

        if hasattr(self, "train_missile") and self.train_missile and self.player:
            if not hasattr(self.player, "missiles") or len(self.player.missiles) == 0:
                self.player.shoot_missile()

            if self.enemy:
                enemy_x, enemy_y = self.enemy.pos["x"], self.enemy.pos["y"]
                self.player.update_missiles((enemy_x, enemy_y))

            px = self.player.position["x"]
            py = self.player.position["y"]
            ex = self.enemy.pos["x"]
            ey = self.enemy.pos["y"]

            action_dx, action_dy = compute_normalized_direction(px, py, ex, ey)
            speed = self.enemy.base_speed
            self.enemy.pos["x"] += action_dx * speed
            self.enemy.pos["y"] += action_dy * speed

            collision = self.check_collision()

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

    # Removed handle_respawn and respawn_enemy methods
    # since their logic has been modularized and we no longer keep spawning code here.

    def check_missile_collisions(self) -> None:
        """
        Check for collisions between missiles and the enemy.
        """
        if not self.enemy or not self.player or not self.enemy.visible:
            return

        enemy_rect = pygame.Rect(
            self.enemy.pos["x"],
            self.enemy.pos["y"],
            self.enemy.size,
            self.enemy.size,
        )

        for missile in self.player.missiles[:]:
            if missile.get_rect().colliderect(enemy_rect):
                logging.info("Missile hit the enemy.")
                self.player.missiles.remove(missile)
                self.enemy.hide()
                self.is_respawning = True
                self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
                logging.info(
                    f"Enemy will respawn in {self.respawn_delay} ms due to missile hit."
                )
