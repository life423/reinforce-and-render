# ai_platform_trainer/engine/entities/missile.py

import pygame
from ai_platform_trainer.core.color_manager import get_color

class Missile:
    def __init__(
        self,
        position: tuple[int, int],
        direction: pygame.math.Vector2,
        speed: float = 8.0,
        radius: int = 5
    ):
        # Position and movement
        self.position = pygame.math.Vector2(position)
        self.direction = direction.normalize()
        self.speed = speed

        # Appearance
        self.color = get_color("accent")
        self.radius = radius

    def update(self, dt: float = 1.0) -> None:
        """
        Move the missile along its direction.
        dt is the delta time factor (defaults to 1 for fixed-step).
        """
        self.position += self.direction * self.speed * dt

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the missile as a filled circle.
        """
        pygame.draw.circle(
            surface,
            self.color,
            (int(self.position.x), int(self.position.y)),
            self.radius
        )
