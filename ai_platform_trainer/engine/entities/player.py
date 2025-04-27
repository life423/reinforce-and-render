# ai_platform_trainer/engine/entities/player.py

from typing import Dict, Tuple
import pygame
import pymunk

class Player:
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int] = None, speed: int = 5):
        self.position = position
        self.speed = speed
        self.color = color if color is not None else (0, 255, 0)  # Use provided color or default to green
        self.radius = 15
        self.body = None  # Will be set when physics body is created
        self.force_scale = 5000.0  # Scale factor for applying forces
        
    def set_physics_body(self, body: pymunk.Body) -> None:
        """
        Set the physics body associated with this player.
        
        Args:
            body: Pymunk physics body
        """
        self.body = body
        
    def update(self, actions: Dict[str, bool]) -> None:
        """
        Apply forces to the physics body based on input actions.
        
        Args:
            actions: Dictionary of input actions
        """
        if not self.body:
            return
            
        # Reset forces
        self.body.force = (0, 0)
        
        # Apply forces based on input
        if actions.get('up'):
            self.body.force += (0, -self.force_scale)
        if actions.get('down'):
            self.body.force += (0, self.force_scale)
        if actions.get('left'):
            self.body.force += (-self.force_scale, 0)
        if actions.get('right'):
            self.body.force += (self.force_scale, 0)
        
        # Update position from physics body
        if self.body:
            self.position = self.body.position
            
    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the player as a circle.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw a circle at the current physics body position
        pos = self.body.position if self.body else self.position
        pygame.draw.circle(surface, self.color, (int(pos[0]), int(pos[1])), self.radius)
