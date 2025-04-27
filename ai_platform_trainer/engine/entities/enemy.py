import random
from typing import Tuple

import pygame


class Enemy:
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int], radius: int = 10):
        # Store position as a single attribute
        self.position = position
        self.color = color
        self.radius = radius
        self.speed = random.randint(1, 3)
        
    def update(self):
        # Now this will work correctly
        x, y = self.position
        
        # Simple movement logic (example)
        dx = random.randint(-self.speed, self.speed)
        dy = random.randint(-self.speed, self.speed)
        
        new_x = max(0, min(800, x + dx))
        new_y = max(0, min(600, y + dy))
        
        self.position = (new_x, new_y)
        
    def draw(self, surface: pygame.Surface):
        """
        Draw the enemy as a circle on the given surface.
        """
        pygame.draw.circle(surface, self.color, self.position, self.radius)
