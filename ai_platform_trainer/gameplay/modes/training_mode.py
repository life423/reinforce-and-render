# training_mode.py
import math
import logging
from ai_platform_trainer.gameplay.utils import compute_normalized_direction
from ai_platform_trainer.gameplay.spawner import respawn_enemy_with_fade_in
import pygame


class TrainingModeManager:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0

    def update(self) -> None:
        # Update player position
        if self.game.enemy and self.game.player:
            self.game.player.update(self.game.enemy.pos["x"], self.game.enemy.pos["y"])

        # Missile firing logic (if train_missile is enabled)
        if (
            hasattr(self.game, "train_missile")
            and self.game.train_missile
            and self.game.player
        ):
            if self.missile_cooldown > 0:
                self.missile_cooldown -= 1
            if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                self.game.player.shoot_missile()
                self.missile_cooldown = 120

        # Update missiles (no homing yet, just straight movement as before)
        self.game.player.update_missiles()

        # Compute enemy movement
        px = self.game.player.position["x"]
        py = self.game.player.position["y"]
        ex = self.game.enemy.pos["x"]
        ey = self.game.enemy.pos["y"]

        action_dx, action_dy = compute_normalized_direction(px, py, ex, ey)
        speed = self.game.enemy.base_speed
        self.game.enemy.pos["x"] += action_dx * speed
        self.game.enemy.pos["y"] += action_dy * speed

        # Check collisions
        collision = self.game.check_collision()

        # Check for missile-enemy collision
        missile_collision = False
        if self.game.enemy and self.game.player and self.game.enemy.visible:
            enemy_rect = pygame.Rect(ex, ey, self.game.enemy.size, self.game.enemy.size)
            for missile in self.game.player.missiles[:]:
                if missile.get_rect().colliderect(enemy_rect):
                    logging.info("Missile hit the enemy (training mode).")
                    self.game.player.missiles.remove(missile)
                    missile_collision = True
                    # Hide enemy and trigger respawn
                    self.game.enemy.hide()
                    self.game.is_respawning = True
                    self.game.respawn_timer = (
                        pygame.time.get_ticks() + self.game.respawn_delay
                    )
                    break

        # If player-enemy collision occurs, handle hide and respawn
        if collision:
            logging.info(
                "Collision detected between player and enemy in training mode."
            )
            self.game.enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = pygame.time.get_ticks() + self.game.respawn_delay

        # If respawning needed, handle it
        current_time = pygame.time.get_ticks()
        if (
            self.game.is_respawning
            and current_time >= self.game.respawn_timer
            and self.game.enemy
            and self.game.player
        ):
            respawn_enemy_with_fade_in(self.game, current_time)

        # Log data
        if self.game.data_logger:
            data_point = {
                "mode": "train",
                "player_x": px,
                "player_y": py,
                "enemy_x": self.game.enemy.pos["x"],
                "enemy_y": self.game.enemy.pos["y"],
                "action_dx": action_dx,
                "action_dy": action_dy,
                "collision": collision,
                "dist": math.hypot(
                    px - self.game.enemy.pos["x"], py - self.game.enemy.pos["y"]
                ),
                "missile_collision": missile_collision,
            }
            self.game.data_logger.log(data_point)
            logging.debug("Logged training data point with collision info.")
