# ai_platform_trainer/engine/entities/entity.py
from typing import Tuple
import pygame

class Entity:
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int], radius: int = 10):
        self.x, self.y = position
        self.color = color
        self.radius = radius

    def update(self, actions: dict) -> None:
        """Override in subclasses to change position each frame."""
        pass

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)
