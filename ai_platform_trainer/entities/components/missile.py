"""
Missile entity for AI Platform Trainer.

This module defines the Missile class, which represents projectiles
that can be fired by both the player and enemies.
"""
import logging
import math
import os

import pygame


class Missile:
    def __init__(
        self,
        x: int,
        y: int,
        speed: float = 5.0,
        vx: float = 5.0,
        vy: float = 0.0,
        birth_time: int = 0,
        lifespan: int = 2000,  # default 2s if not overridden
    ):
        self.size = 10
        self.color = (255, 255, 0)  # Yellow fallback color
        self.pos = {"x": x, "y": y}
        self.speed = speed
        # Velocity components for straight line movement
        self.vx = vx
        self.vy = vy

        # New fields for matching training logic:
        self.birth_time = birth_time
        self.lifespan = lifespan
        
        # Calculate initial rotation angle based on velocity
        self.angle = math.degrees(math.atan2(vy, vx))
        
        # Load sprite
        self.sprite = self._load_sprite()
        self.using_sprite = self.sprite is not None
        
        # Visual effects
        self.trail_positions = []
        self.max_trail_length = 5
        self.trail_color = (255, 140, 0)  # Orange trail

    def _load_sprite(self) -> pygame.Surface:
        """Load the missile sprite from assets."""
        try:
            sprite_path = os.path.join("assets", "sprites", "missile", "missile.png")
            sprite = pygame.image.load(sprite_path)
            # Scale missile to appropriate size
            return pygame.transform.scale(sprite, (self.size * 3, self.size))
        except (pygame.error, FileNotFoundError) as e:
            logging.error(f"Could not load missile sprite: {e}")
            return None

    def update(self) -> None:
        """
        Update missile position based on its velocity.
        """
        # Add current position to trail before updating
        if len(self.trail_positions) >= self.max_trail_length:
            self.trail_positions.pop(0)
        self.trail_positions.append((int(self.pos["x"]), int(self.pos["y"])))
        
        # Update position
        self.pos["x"] += self.vx
        self.pos["y"] += self.vy

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the missile on the screen with visual effects."""
        # Draw trail
        for i, pos in enumerate(self.trail_positions):
            # Fade trail from start to end
            alpha = int(255 * (i + 1) / len(self.trail_positions)) 
            radius = int(self.size * 0.6 * (i + 1) / len(self.trail_positions))
            
            # Create a surface for the trail particle
            trail_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                trail_surface,
                (*self.trail_color, alpha),
                (radius, radius),
                radius
            )
            screen.blit(trail_surface, (pos[0] - radius, pos[1] - radius))
        
        if self.using_sprite:
            # Rotate sprite to match direction
            rotated_sprite = pygame.transform.rotate(self.sprite, -self.angle)
            # Get the rect of the rotated sprite to ensure it's centered
            rect = rotated_sprite.get_rect(center=(int(self.pos["x"]), int(self.pos["y"])))
            screen.blit(rotated_sprite, rect)
        else:
            # Fallback to circle if sprite isn't available
            pygame.draw.circle(
                screen,
                self.color,
                (int(self.pos["x"]), int(self.pos["y"])),
                self.size,
            )

    def get_rect(self) -> pygame.Rect:
        """Get the missile's rectangle for collision detection."""
        if self.using_sprite:
            # For sprite-based missile, use a rectangle that matches the sprite
            size_x = self.size * 3
            size_y = self.size
            return pygame.Rect(
                self.pos["x"] - size_x // 2,
                self.pos["y"] - size_y // 2,
                size_x,
                size_y,
            )
        else:
            # For circle-based missile, use a square that encompasses the circle
            return pygame.Rect(
                self.pos["x"] - self.size,
                self.pos["y"] - self.size,
                self.size * 2,
                self.size * 2,
            )
