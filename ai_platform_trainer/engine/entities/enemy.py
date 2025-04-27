import random
from typing import Tuple

import pygame
import pymunk


class Enemy:
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int], radius: int = 10):
        # Store position as a single attribute
        self.position = position
        self.color = color
        self.radius = radius
        self.body = None  # Will be set when physics body is created
        
        # Apply random initial impulse
        self.initial_impulse = (
            random.uniform(-100, 100),
            random.uniform(-100, 100)
        )
        
    def set_physics_body(self, body: pymunk.Body) -> None:
        """
        Set the physics body associated with this enemy.
        
        Args:
            body: Pymunk physics body
        """
        self.body = body
        
        # Apply initial impulse to get movement started
        if self.body:
            self.body.apply_impulse_at_local_point(self.initial_impulse)
        
    def update(self) -> None:
        """Update the enemy position from its physics body."""
        if self.body:
            self.position = self.body.position
            
            # Occasionally apply a small random impulse to keep movement varied
            if random.random() < 0.02:  # 2% chance per frame
                random_impulse = (
                    random.uniform(-50, 50),
                    random.uniform(-50, 50)
                )
                self.body.apply_impulse_at_local_point(random_impulse)
        
    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the enemy as a circle on the given surface.
        """
        # Draw a circle at the current physics body position
        pos = self.body.position if self.body else self.position
        pygame.draw.circle(surface, self.color, (int(pos[0]), int(pos[1])), self.radius)
