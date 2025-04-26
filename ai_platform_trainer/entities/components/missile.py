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
        # Store the actual display dimensions (missile is elongated)
        self.width = self.size * 3
        self.height = self.size * 1.5
        self.color = (255, 255, 0)  # Yellow fallback color
        
        # Position is the CENTER of the missile, not top-left corner
        self.position = {"x": x, "y": y}
        # Also maintain pos for compatibility with any code that uses it
        self.pos = self.position
        
        self.speed = speed
        # Velocity components for straight line movement
        self.vx = vx
        self.vy = vy
        # Direction vector for renderer
        self.direction = (vx, vy)

        # New fields for matching training logic:
        self.birth_time = birth_time
        self.lifespan = lifespan
        
        # Calculate initial rotation angle based on velocity
        self.angle = math.degrees(math.atan2(vy, vx))
        # Add debug flag
        self.show_debug = True  # Set to False in production
        
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
            # Use the standardized dimensions
            return pygame.transform.scale(sprite, (int(self.width), int(self.height)))
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
        logging.debug(
            f"Drawing missile: pos=({self.position['x']}, {self.position['y']}), "
            f"angle={self.angle:.1f}°, sprite={'available' if self.using_sprite else 'unavailable'}"
        )
        
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
            rect = rotated_sprite.get_rect(
                center=(int(self.position["x"]), int(self.position["y"]))
            )
            
            logging.debug(
                f"Sprite details: original_size={self.sprite.get_size()}, "
                f"rotated_size={rotated_sprite.get_size()}, "
                f"draw_rect={rect}"
            )
            
            screen.blit(rotated_sprite, rect)
            
            # Draw debug visualization if enabled
            if self.show_debug:
                # Draw center point
                pygame.draw.circle(
                    screen,
                    (255, 0, 0),  # Red for center
                    (int(self.position["x"]), int(self.position["y"])),
                    3
                )
                
                # Draw bounding box
                rect = self.get_rect()
                pygame.draw.rect(
                    screen,
                    (0, 255, 0),  # Green for hitbox
                    rect,
                    1  # Line width
                )
                
                # Draw direction line
                line_length = 20
                end_x = int(self.position["x"] + line_length * math.cos(math.radians(self.angle)))
                end_y = int(self.position["y"] + line_length * math.sin(math.radians(self.angle)))
                pygame.draw.line(
                    screen,
                    (0, 0, 255),  # Blue for direction
                    (int(self.position["x"]), int(self.position["y"])),
                    (end_x, end_y),
                    2
                )
        else:
            # Fallback to circle if sprite isn't available
            logging.warning("Using fallback circle for missile - sprite not available")
            pygame.draw.circle(
                screen,
                self.color,
                (int(self.position["x"]), int(self.position["y"])),
                self.size,
            )

    def get_rect(self) -> pygame.Rect:
        """Get the missile's rectangle for collision detection."""
        if self.using_sprite:
            # Center the rectangle on the missile's position
            return pygame.Rect(
                self.position["x"] - self.width // 2,
                self.position["y"] - self.height // 2,
                self.width,
                self.height
            )
        else:
            # For circle-based missile, use a square that encompasses the circle
            return pygame.Rect(
                self.position["x"] - self.size,
                self.position["y"] - self.size,
                self.size * 2,
                self.size * 2,
            )
