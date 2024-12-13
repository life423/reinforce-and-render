import pygame
import math
import logging
from typing import Tuple  # Add this import


class Missile:
    def __init__(self, x: int, y: int, speed: float = 5.0):
        self.size = 10
        self.color = (255, 255, 0)  # Yellow
        self.pos = {"x": x, "y": y}
        self.speed = speed
        self.velocity = {"x": 0.0, "y": 0.0}

    def update(self, target_pos: Tuple[int, int]) -> None:
        """
        Update missile position and adjust direction towards the target.

        :param target_pos: Tuple containing target's (x, y) position
        """
        target_x, target_y = target_pos
        direction_x = target_x - self.pos["x"]
        direction_y = target_y - self.pos["y"]
        distance = math.hypot(direction_x, direction_y)
        if distance > 0:
            self.velocity["x"] = (direction_x / distance) * self.speed
            self.velocity["y"] = (direction_y / distance) * self.speed
        else:
            self.velocity["x"] = 0.0
            self.velocity["y"] = 0.0

        self.pos["x"] += self.velocity["x"]
        self.pos["y"] += self.velocity["y"]

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the missile on the screen."""
        pygame.draw.circle(
            screen,
            self.color,
            (int(self.pos["x"]), int(self.pos["y"])),
            self.size,
        )

    def get_rect(self) -> pygame.Rect:
        """Get the missile's rectangle for collision detection."""
        return pygame.Rect(
            self.pos["x"] - self.size,
            self.pos["y"] - self.size,
            self.size * 2,
            self.size * 2,
        )
