# ai_platform_trainer/entities/player_training.py
import random
import pygame
import logging
import math
from ai_platform_trainer.entities.missile import Missile


class PlayerTraining:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.step = 5
        self.missiles = []
        logging.info("PlayerTraining initialized.")

    def reset(self) -> None:
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.missiles.clear()
        logging.info("PlayerTraining has been reset.")

    def update(self, enemy_x: float, enemy_y: float) -> None:
        """
        Update the player's position each frame during training mode.
        The player will maintain a good distance from the enemy:
        - If too close, move directly away.
        - If at a safe distance, remain still.

        Wrap-around logic is handled by modulo arithmetic to ensure stable wrapping.
        """

        desired_distance = 200
        px, py = self.position["x"], self.position["y"]
        dx = px - enemy_x
        dy = py - enemy_y
        dist = math.hypot(dx, dy)

        if dist < desired_distance:
            # Enemy is too close, move away directly
            if dist > 0:
                ndx, ndy = dx / dist, dy / dist
            else:
                # Exactly overlapping enemy; pick a random direction to escape
                ndx, ndy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            # Move away at normal speed
            self.position["x"] += ndx * self.step
            self.position["y"] += ndy * self.step
        else:
            # Enemy is far enough; remain still
            pass

        # Wrap-around using modulo to ensure stable wrapping
        self.position["x"] %= self.screen_width
        self.position["y"] %= self.screen_height

    def shoot_missile(self) -> None:
        missile_start_x = self.position["x"] + self.size // 2
        missile_start_y = self.position["y"] + self.size // 2
        missile = Missile(x=missile_start_x, y=missile_start_y, vx=5.0, vy=0.0)
        self.missiles.append(missile)
        logging.info("Training mode: Missile shot straight to the right.")

    def update_missiles(self, enemy_pos: tuple[int, int]) -> None:
        ex, ey = enemy_pos
        for missile in self.missiles[:]:
            missile.update()
            # Remove missile if it goes off-screen
            # If you want missiles to wrap as well, you can also apply modulo here
            if (
                missile.pos["x"] < 0
                or missile.pos["x"] > self.screen_width
                or missile.pos["y"] < 0
                or missile.pos["y"] > self.screen_height
            ):
                self.missiles.remove(missile)
                logging.debug("Missile removed for going off-screen.")

    def draw_missiles(self, screen: pygame.Surface) -> None:
        for missile in self.missiles:
            missile.draw(screen)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
        self.draw_missiles(screen)
