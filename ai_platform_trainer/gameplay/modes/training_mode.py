# training_mode.py
import math
import logging
import pygame
from ai_platform_trainer.gameplay.utils import compute_normalized_direction
from ai_platform_trainer.gameplay.spawner import respawn_enemy_with_fade_in


class TrainingModeManager:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_turn_rate = math.radians(
            5
        )  # Allowing slight steering angle per update

    def update(self) -> None:
        # Update player position relative to enemy
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

        # Initialize missile tracking variables for logging
        missile_x, missile_y = None, None
        missile_angle = None
        missile_action = 0.0
        missile_collision = False

        # Homing missile logic if a missile is present
        if self.game.player and self.game.player.missiles:
            missile = self.game.player.missiles[0]

            # Homing logic - steer missile slightly towards enemy
            ex = self.game.enemy.pos["x"]
            ey = self.game.enemy.pos["y"]
            mx, my = missile.pos["x"], missile.pos["y"]
            dx, dy = ex - mx, ey - my

            current_angle = math.atan2(missile.vy, missile.vx)
            desired_angle = math.atan2(dy, dx)
            angle_diff = (desired_angle - current_angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi

            # Limit angle change to missile_turn_rate
            if angle_diff > self.missile_turn_rate:
                angle_diff = self.missile_turn_rate
            elif angle_diff < -self.missile_turn_rate:
                angle_diff = -self.missile_turn_rate

            missile_action = angle_diff
            new_angle = current_angle + angle_diff

            # Update missile velocity for homing
            speed = 5.0
            missile.vx = math.cos(new_angle) * speed
            missile.vy = math.sin(new_angle) * speed
            missile.update()

            missile_x = missile.pos["x"]
            missile_y = missile.pos["y"]
            missile_angle = new_angle
        else:
            # If no missile present, just update missiles normally (if any exist)
            if self.game.player:
                self.game.player.update_missiles()

        # Compute enemy movement (simple chase)
        px = self.game.player.position["x"] if self.game.player else 0
        py = self.game.player.position["y"] if self.game.player else 0
        ex = self.game.enemy.pos["x"] if self.game.enemy else 0
        ey = self.game.enemy.pos["y"] if self.game.enemy else 0

        action_dx, action_dy = compute_normalized_direction(px, py, ex, ey)
        if self.game.enemy:
            speed = self.game.enemy.base_speed
            self.game.enemy.pos["x"] += action_dx * speed
            self.game.enemy.pos["y"] += action_dy * speed

        # Check player-enemy collision
        collision = self.game.check_collision()
        if collision:
            logging.info(
                "Collision detected between player and enemy in training mode."
            )
            self.game.enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = pygame.time.get_ticks() + self.game.respawn_delay

        # Check missile-enemy collision for training mode
        if (
            self.game.enemy
            and self.game.player
            and self.game.enemy.visible
            and self.game.player.missiles
        ):
            enemy_rect = pygame.Rect(ex, ey, self.game.enemy.size, self.game.enemy.size)
            for msl in self.game.player.missiles[:]:
                if msl.get_rect().colliderect(enemy_rect):
                    logging.info("Missile hit the enemy (training mode).")
                    self.game.player.missiles.remove(msl)
                    self.game.enemy.hide()
                    self.game.is_respawning = True
                    self.game.respawn_timer = (
                        pygame.time.get_ticks() + self.game.respawn_delay
                    )
                    missile_collision = True
                    break

        # Handle respawn if needed
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
                "enemy_x": ex,
                "enemy_y": ey,
                "action_dx": action_dx,
                "action_dy": action_dy,
                "collision": collision,
                "dist": math.hypot(px - ex, py - ey),
                "missile_collision": missile_collision,
            }

            if missile_x is not None and missile_y is not None:
                data_point["missile_x"] = missile_x
                data_point["missile_y"] = missile_y
                data_point["missile_angle"] = missile_angle
                data_point["missile_action"] = missile_action

            self.game.data_logger.log(data_point)
            logging.debug(
                "Logged training data point with missile homing and collision info."
            )
