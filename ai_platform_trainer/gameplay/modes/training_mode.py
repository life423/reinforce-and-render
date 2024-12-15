# training_mode.py
import math
import logging
import random
from ai_platform_trainer.gameplay.utils import compute_normalized_direction


class TrainingModeManager:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0  # cooldown frames before next missile
        self.missile_turn_rate = math.radians(5)  # max angle change per update

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

        # Update missiles (basic homing logic)
        missile_angle = None
        missile_action = 0.0
        missile_x = None
        missile_y = None

        if self.game.player and self.game.player.missiles:
            # Assume one missile at a time for simplicity
            missile = self.game.player.missiles[0]

            # Current missile position
            missile_x = missile.pos["x"]
            missile_y = missile.pos["y"]

            # Compute direction from missile to enemy
            ex = self.game.enemy.pos["x"]
            ey = self.game.enemy.pos["y"]
            mx, my = missile_x, missile_y
            dx = ex - mx
            dy = ey - my

            # Current missile angle based on vx, vy
            current_angle = math.atan2(missile.vy, missile.vx)

            # Desired angle towards enemy
            desired_angle = math.atan2(dy, dx)

            # Compute smallest angle difference
            angle_diff = (desired_angle - current_angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi

            # Limit angle change by missile_turn_rate
            if angle_diff > self.missile_turn_rate:
                angle_diff = self.missile_turn_rate
            elif angle_diff < -self.missile_turn_rate:
                angle_diff = -self.missile_turn_rate

            # Apply angle change
            missile_action = angle_diff
            new_angle = current_angle + angle_diff

            # Update missile velocity based on new angle
            speed = 5.0  # missile speed is constant, or adjust as needed
            missile.vx = math.cos(new_angle) * speed
            missile.vy = math.sin(new_angle) * speed

            # After steering, update missile position
            missile.update()

            # Update missile_x,y and angle after update
            missile_x = missile.pos["x"]
            missile_y = missile.pos["y"]
            missile_angle = new_angle

        # Update enemy movement towards player
        if self.game.enemy and self.game.player:
            px = self.game.player.position["x"]
            py = self.game.player.position["y"]
            ex = self.game.enemy.pos["x"]
            ey = self.game.enemy.pos["y"]

            action_dx, action_dy = compute_normalized_direction(px, py, ex, ey)
            speed = self.game.enemy.base_speed
            self.game.enemy.pos["x"] += action_dx * speed
            self.game.enemy.pos["y"] += action_dy * speed
        else:
            px = py = ex = ey = 0
            action_dx = action_dy = 0.0

        # Check for collision (player-enemy)
        collision = self.game.check_collision()

        # Log data, including missile info if available
        if self.game.data_logger:
            data_point = {
                "mode": "train",
                "player_x": px,
                "player_y": py,
                "enemy_x": self.game.enemy.pos["x"] if self.game.enemy else None,
                "enemy_y": self.game.enemy.pos["y"] if self.game.enemy else None,
                "action_dx": action_dx,
                "action_dy": action_dy,
                "collision": collision,
                "dist": math.hypot(px - ex, py - ey),
            }

            # Add missile data if a missile is present
            if missile_x is not None and missile_y is not None:
                data_point["missile_x"] = missile_x
                data_point["missile_y"] = missile_y
                data_point["missile_angle"] = missile_angle
                data_point["missile_action"] = missile_action

            self.game.data_logger.log(data_point)
            logging.debug("Logged training data point with missile info.")
