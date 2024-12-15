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
        self.missile_turn_rate = math.radians(3)  # Reduced turn rate
        self.missile_lifespan = {}  # missile -> (birth_time, lifespan)

    def update(self) -> None:
        # Update player relative to the enemy
        if self.game.enemy and self.game.player:
            self.game.player.update(self.game.enemy.pos["x"], self.game.enemy.pos["y"])

        # Missile firing logic
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

                # Assign random lifespan (0.5s to 1.5s)
                if self.game.player.missiles:
                    msl = self.game.player.missiles[0]
                    birth_time = pygame.time.get_ticks()
                    lifespan = random.randint(500, 1500)
                    self.missile_lifespan[msl] = (birth_time, lifespan)

        missile_x, missile_y = None, None
        missile_angle = None
        missile_action = 0.0
        missile_collision = False
        current_time = pygame.time.get_ticks()

        # Update and check missile lifespan
        for msl in self.game.player.missiles[:]:
            if msl in self.missile_lifespan:
                birth, lifespan = self.missile_lifespan[msl]
                if current_time - birth >= lifespan:
                    # Missile expired
                    self.game.player.missiles.remove(msl)
                    del self.missile_lifespan[msl]
                    continue

            if self.game.enemy and self.game.enemy.visible:
                # Homing with noise and occasional missed turns
                ex = self.game.enemy.pos["x"]
                ey = self.game.enemy.pos["y"]
                mx, my = msl.pos["x"], msl.pos["y"]
                dx, dy = ex - mx, ey - my

                current_angle = math.atan2(msl.vy, msl.vx)
                desired_angle = math.atan2(dy, dx)
                angle_diff = (desired_angle - current_angle) % (2 * math.pi)
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi

                # Add random noise
                noise = random.uniform(-math.radians(5), math.radians(5))
                angle_diff += noise

                # 10% chance of no turn
                if random.random() < 0.1:
                    angle_diff = 0.0

                # Limit angle change
                if angle_diff > self.missile_turn_rate:
                    angle_diff = self.missile_turn_rate
                elif angle_diff < -self.missile_turn_rate:
                    angle_diff = -self.missile_turn_rate

                missile_action = angle_diff
                new_angle = current_angle + angle_diff

                # Add slight sinusoidal offset for weaving
                time_factor = current_time * 0.002
                sine_offset = math.sin(time_factor) * math.radians(3)
                new_angle += sine_offset

                # Randomize speed slightly
                speed = 5.0 * random.uniform(0.8, 1.0)
                msl.vx = math.cos(new_angle) * speed
                msl.vy = math.sin(new_angle) * speed
                msl.update()

                missile_x = msl.pos["x"]
                missile_y = msl.pos["y"]
                missile_angle = new_angle
            else:
                # No enemy or not visible, no homing
                msl.update()

        # Enemy movement (simple chase)
        px = self.game.player.position["x"] if self.game.player else 0
        py = self.game.player.position["y"] if self.game.player else 0
        ex = self.game.enemy.pos["x"] if self.game.enemy else 0
        ey = self.game.enemy.pos["y"] if self.game.enemy else 0

        action_dx, action_dy = compute_normalized_direction(px, py, ex, ey)
        if self.game.enemy:
            speed = self.game.enemy.base_speed
            self.game.enemy.pos["x"] += action_dx * speed
            self.game.enemy.pos["y"] += action_dy * speed

        # Check collisions
        collision = self.game.check_collision()
        if collision:
            logging.info(
                "Collision detected between player and enemy in training mode."
            )
            self.game.enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay

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
                    if msl in self.missile_lifespan:
                        del self.missile_lifespan[msl]
                    self.game.enemy.hide()
                    self.game.is_respawning = True
                    self.game.respawn_timer = current_time + self.game.respawn_delay
                    missile_collision = True
                    break

        # Handle respawn
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
                "Logged training data point with reduced missile accuracy, sinusoidal weaving, and smoother player movement."
            )
