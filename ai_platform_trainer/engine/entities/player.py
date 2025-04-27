# ai_platform_trainer/engine/entities/player.py

from typing import Dict, Tuple

import pygame


class Player:
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int] = None, speed: int = 5):
        self.position = position
        self.speed = speed
        self.color = color if color is not None else (0, 255, 0)  # Use provided color or default to green
        self.radius = 15
        
    def update(self, actions: Dict[str, bool]):
        x, y = self.position
        
        if actions.get('up'):
            y -= self.speed
        if actions.get('down'):
            y += self.speed
        if actions.get('left'):
            x -= self.speed
        if actions.get('right'):
            x += self.speed
            
        # Ensure player stays within screen boundaries
        x = max(0, min(800, x))
        y = max(0, min(600, y))
        
        self.position = (x, y)
        
    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, self.color, self.position, self.radius)
