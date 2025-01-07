import math
import logging
import random
import pygame
from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.utils.helpers import wrap_position
from ai_platform_trainer.gameplay.utils import compute_normalized_direction
from ai_platform_trainer.gameplay.spawner import respawn_enemy_with_fade_in


class TrainingMode:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_turn_rate = math.radians(3)  # Reduced turn rate
        self.missile_lifespan = {}  # missile -> (birth_time, lifespan)

        # NEW: track each missile's entire frame-by-frame data
        # until collision or expiry
        self.missile_sequences = {}  # missile -> [list of data_points]

    def update(self) -> None:
        # 1) Update player
        if self.game.enemy and self.game.player:
            self.game.player.update(self.game.enemy.pos["x"], self.game.enemy.pos["y"])

        # 2) Possibly shoot a missile if none exist, with a random lifespan
        if getattr(self.game, "train_missile", False) and self.game.player:
            if self.missile_cooldown > 0:
                self.missile_cooldown -= 1
            if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                self.game.player.shoot_missile()
                self.missile_cooldown = 120

                if self.game.player.missiles:
                    msl = self.game.player.missiles[0]
                    birth_time = pygame.time.get_ticks()
                    lifespan = random.randint(500, 1500)
                    self.missile_lifespan[msl] = (birth_time, lifespan)

                    # Initialize empty list for this missile's frames
                    self.missile_sequences[msl] = []

        missile_x, missile_y = None, None
        missile_angle = None
        missile_action = 0.0
        missile_collision = False
        current_time = pygame.time.get_ticks()

        # 3) Update existing missiles
        for msl in self.game.player.missiles[:]:
            # Check missile lifespan
            if msl in self.missile_lifespan:
                birth, lifespan = self.missile_lifespan[msl]
                if current_time - birth >= lifespan:
                    # MISSILE EXPIRED => log sequence as outcome=0
                    self.finalize_missile_sequence(msl, success=False)
                    # remove from lists
                    self.game.player.missiles.remove(msl)
                    del self.missile_lifespan[msl]
                    continue

            if self.game.enemy and self.game.enemy.visible:
                # HOMING with random arcs
                ex = self.game.enemy.pos["x"]
                ey = self.game.enemy.pos["y"]
                mx, my = msl.pos["x"], msl.pos["y"]
                dx, dy = ex - mx, ey - my

                current_angle = math.atan2(msl.vy, msl.vx)
                desired_angle = math.atan2(dy, dx)
                angle_diff = (desired_angle - current_angle) % (2 * math.pi)
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi

                # Random noise
                noise = random.uniform(-math.radians(1), math.radians(1))
                angle_diff += noise

                # 10% chance of no turn
                if random.random() < 0.0:
                    angle_diff = 0.0

                # Limit angle change
                if angle_diff > self.missile_turn_rate:
                    angle_diff = self.missile_turn_rate
                elif angle_diff < -self.missile_turn_rate:
                    angle_diff = -self.missile_turn_rate

                missile_action = angle_diff
                new_angle = current_angle + angle_diff

                # slight sinusoidal offset for weaving
                time_factor = current_time * 0.002
                # sine_offset = math.sin(time_factor) * math.radians(3)
                # new_angle += sine_offset

                # random speed
                speed = 5.0
                msl.vx = math.cos(new_angle) * speed
                msl.vy = math.sin(new_angle) * speed
                msl.update()

                missile_x = msl.pos["x"]
                missile_y = msl.pos["y"]
                missile_angle = new_angle
            else:
                # No enemy or not visible => no homing
                msl.update()

            # Build a single frame data point
            px = self.game.player.position["x"] if self.game.player else 0
            py = self.game.player.position["y"] if self.game.player else 0
            ex = self.game.enemy.pos["x"] if self.game.enemy else 0
            ey = self.game.enemy.pos["y"] if self.game.enemy else 0

            # Fix: replace None with 0.0
            if missile_x is None:
                missile_x = 0.0
            if missile_y is None:
                missile_y = 0.0
            if missile_angle is None:
                missile_angle = 0.0

            data_point = {
                "mode": "train",
                "player_x": px,
                "player_y": py,
                "enemy_x": ex,
                "enemy_y": ey,
                "dist": math.hypot(px - ex, py - ey),
                "missile_x": missile_x,
                "missile_y": missile_y,
                "missile_angle": missile_angle,
                "missile_action": missile_action,
                "missile_collision": False,  # default, update on collision
            }

            # store this frame into self.missile_sequences[msl]
            if msl in self.missile_sequences:
                self.missile_sequences[msl].append(data_point)

        # 4) Enemy "simple chase"
        if self.game.enemy and self.game.player:
            px = self.game.player.position["x"]
            py = self.game.player.position["y"]
            ex = self.game.enemy.pos["x"]
            ey = self.game.enemy.pos["y"]
            action_dx, action_dy = compute_normalized_direction(px, py, ex, ey)
            speed = self.game.enemy.base_speed
            self.game.enemy.pos["x"] += action_dx * speed
            self.game.enemy.pos["y"] += action_dy * speed

        # 5) Check collisions => if collision => finalize
        collision = self.game.check_collision()
        if collision:
            logging.info("Collision between player and enemy in training mode.")
            self.game.enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay

        # 6) Check missile => enemy collisions
        if (
            self.game.enemy
            and self.game.player
            and self.game.enemy.visible
            and self.game.player.missiles
        ):
            ex = self.game.enemy.pos["x"]
            ey = self.game.enemy.pos["y"]
            enemy_rect = pygame.Rect(ex, ey, self.game.enemy.size, self.game.enemy.size)
            for msl in self.game.player.missiles[:]:
                if msl.get_rect().colliderect(enemy_rect):
                    logging.info("Missile hit the enemy (training mode).")
                    # finalize as success=1
                    self.finalize_missile_sequence(msl, success=True)
                    self.game.player.missiles.remove(msl)
                    if msl in self.missile_lifespan:
                        del self.missile_lifespan[msl]
                    self.game.enemy.hide()
                    self.game.is_respawning = True
                    self.game.respawn_timer = current_time + self.game.respawn_delay
                    break

        # 7) Handle enemy respawn
        if (
            self.game.is_respawning
            and current_time >= self.game.respawn_timer
            and self.game.enemy
            and self.game.player
        ):
            respawn_enemy_with_fade_in(self.game, current_time)

    def finalize_missile_sequence(self, missile, success: bool) -> None:
        """
        Mark all frames in missile's sequence with 'missile_collision'= success
        Then log them to self.game.data_logger, and clear from missile_sequences.
        """
        if missile not in self.missile_sequences:
            return

        outcome_val = success  # True/False => 1.0/0.0 if you prefer
        frames = self.missile_sequences[missile]

        for frame_data in frames:
            frame_data["missile_collision"] = outcome_val

            # now actually log to data_logger
            if self.game.data_logger:
                self.game.data_logger.log(frame_data)

        # remove from dict
        del self.missile_sequences[missile]
        logging.debug(
            f"Finalized missile sequence with success={success}, frames={len(frames)}"
        )
