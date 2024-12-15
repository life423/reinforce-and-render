# training_mode.py
import math
import logging
import random
import pygame
from ai_platform_trainer.gameplay.utils import compute_normalized_direction
from ai_platform_trainer.gameplay.spawner import respawn_enemy_with_fade_in


class TrainingModeManager:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_turn_rate = math.radians(3)  # Reduced turn rate for less accuracy
        # To store missile birth times and lifespans
        # We'll keep a dict: missile -> (birth_time, lifespan)
        self.missile_lifespan = {}

    def update(self) -> None:
        # Update player position relative to the enemy
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

                # Assign a random lifespan (700ms to 1000ms) to the new missile
                if self.game.player.missiles:
                    msl = self.game.player.missiles[0]
                    birth_time = pygame.time.get_ticks()
                    lifespan = random.randint(500, 1500)  # in milliseconds
                    self.missile_lifespan[msl] = (birth_time, lifespan)

        # Initialize missile tracking variables for logging
        missile_x, missile_y = None, None
        missile_angle = None
        missile_action = 0.0
        missile_collision = False

        current_time = pygame.time.get_ticks()

        # Check missile lifespan and homing logic if a missile is present
        # Remove expired missiles
        for msl in self.game.player.missiles[:]:
            if msl in self.missile_lifespan:
                birth, lifespan = self.missile_lifespan[msl]
                if current_time - birth >= lifespan:
                    # Missile expired, remove it
                    self.game.player.missiles.remove(msl)
                    del self.missile_lifespan[msl]
                    continue

            # Homing logic applies only if still present after lifespan check
            if self.game.enemy and self.game.enemy.visible:
                # Homing logic - steer missile slightly towards enemy but add randomness
                ex = self.game.enemy.pos["x"]
                ey = self.game.enemy.pos["y"]
                mx, my = msl.pos["x"], msl.pos["y"]
                dx, dy = ex - mx, ey - my

                current_angle = math.atan2(msl.vy, msl.vx)
                desired_angle = math.atan2(dy, dx)
                angle_diff = (desired_angle - current_angle) % (2 * math.pi)
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi

                # Add a small random noise to angle_diff to make missile less accurate
                noise = random.uniform(-math.radians(5), math.radians(5))
                angle_diff += noise

                # Occasionally skip turning this frame to simulate delay
                if random.random() < 0.1:
                    angle_diff = 0.0

                # Limit angle change
                if angle_diff > self.missile_turn_rate:
                    angle_diff = self.missile_turn_rate
                elif angle_diff < -self.missile_turn_rate:
                    angle_diff = -self.missile_turn_rate

                missile_action = angle_diff
                new_angle = current_angle + angle_diff

                # Slightly randomize speed each update to reduce reliability
                speed = 5.0 * random.uniform(0.8, 1.0)
                msl.vx = math.cos(new_angle) * speed
                msl.vy = math.sin(new_angle) * speed
                msl.update()

                missile_x = msl.pos["x"]
                missile_y = msl.pos["y"]
                missile_angle = new_angle
            else:
                # If no enemy or not visible, just update missiles normally
                msl.update()

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
            self.game.respawn_timer = current_time + self.game.respawn_delay

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
                    # Also remove from lifespan dict if present
                    if msl in self.missile_lifespan:
                        del self.missile_lifespan[msl]
                    self.game.enemy.hide()
                    self.game.is_respawning = True
                    self.game.respawn_timer = current_time + self.game.respawn_delay
                    missile_collision = True
                    break

        # Handle respawn if needed
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
                "Logged training data point with missile lifespan, increased noise, reduced turn rate, and occasional no-turn frames."
            )
