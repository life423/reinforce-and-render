import random
import math
from ai_platform_trainer.entities.player import Player


class PlayerTraining(Player):
    def update(self, enemy_x, enemy_y):
        # Example: random/noise based logic for training mode
        angle = random.uniform(0, 2 * math.pi)
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Move randomly
        self.position["x"] += dx * self.step
        self.position["y"] += dy * self.step
        self.clamp_position()

    # No handle_input here, as training mode doesn't use keyboard input.
