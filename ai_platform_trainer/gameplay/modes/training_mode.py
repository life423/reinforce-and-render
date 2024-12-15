# ai_platform_trainer/gameplay/modes/training_mode.py
import math
import logging
from ai_platform_trainer.gameplay.utils import compute_normalized_direction


class TrainingModeManager:
    def __init__(self, game):
        self.game = game

    def update(self) -> None:
        # Update player position based on enemy position
        self.game.player.update(self.game.enemy.pos["x"], self.game.enemy.pos["y"])

        # Handle missile firing logic in training mode
        if self.game.train_missile:
            # If no missile currently exists, shoot one
            if (
                not hasattr(self.game.player, "missiles")
                or len(self.game.player.missiles) == 0
            ):
                self.game.player.shoot_missile()

            # Update missiles without passing enemy_pos since it's unused
            self.game.player.update_missiles()

        # Compute direction for enemy movement
        px = self.game.player.position["x"]
        py = self.game.player.position["y"]
        ex = self.game.enemy.pos["x"]
        ey = self.game.enemy.pos["y"]

        action_dx, action_dy = compute_normalized_direction(px, py, ex, ey)
        speed = self.game.enemy.base_speed
        self.game.enemy.pos["x"] += action_dx * speed
        self.game.enemy.pos["y"] += action_dy * speed

        # Check for collisions
        collision = self.game.check_collision()

        # Log data if data_logger is available
        if self.game.data_logger:
            self.game.data_logger.log(
                {
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
                }
            )
            logging.debug("Logged training data point.")
