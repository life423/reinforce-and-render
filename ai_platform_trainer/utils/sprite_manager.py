"""
Sprite manager for AI Platform Trainer.

This module handles loading, caching, and rendering sprites for game entities.
It provides an abstraction layer over pygame's sprite handling to make the
code more modular and maintainable.
"""
import logging
import os
from typing import Dict, List, Tuple, Union

import pygame

# Type aliases
Color = Tuple[int, int, int]
Position = Dict[str, float]  # {"x": x, "y": y}


class SpriteManager:
    """
    Manages sprite loading, caching, and rendering.

    This class handles the loading of sprite assets and creates
    placeholder sprites if assets are not available.
    """

    def __init__(self, sprites_dir: str = "assets/sprites"):
        """
        Initialize the sprite manager.

        Args:
            sprites_dir: Directory containing sprite assets
        """
        self.sprites_dir = sprites_dir
        # Log the current working directory to help debug path issues
        cwd = os.getcwd()
        logging.info(f"Current working directory: {cwd}")
        abs_path = os.path.abspath(sprites_dir)
        logging.info(f"Absolute sprite directory path: {abs_path}")
        self.sprites: Dict[str, pygame.Surface] = {}
        self.animations: Dict[str, List[pygame.Surface]] = {}

        # Placeholder colors for entities
        self.placeholder_colors = {
            "player": (0, 128, 255),    # Blue
            "enemy": (255, 50, 50),     # Red
            "missile": (255, 255, 50)   # Yellow
        }

    def load_sprite(self, name: str, size: Tuple[int, int]) -> pygame.Surface:
        """
        Load a sprite image or create a placeholder if not found.

        Args:
            name: Name of the sprite to load
            size: Size (width, height) of the sprite

        Returns:
            Loaded sprite surface or a placeholder
        """
        # Check if sprite is already cached
        if name in self.sprites:
            logging.debug(f"Using cached sprite for '{name}'")
            return self.sprites[name]

        logging.info(f"Loading sprite '{name}' with size {size}")
        
        # Try to load the sprite from subdirectory first (assets/sprites/player/player.png)
        subdirectory_path = os.path.join(self.sprites_dir, name, f"{name}.png")
        logging.debug(f"Checking subdirectory path: {subdirectory_path}")
        
        if os.path.exists(subdirectory_path):
            logging.debug(f"Found sprite at {subdirectory_path}")
            try:
                # Load and scale the sprite
                sprite = pygame.image.load(subdirectory_path).convert_alpha()
                sprite = pygame.transform.scale(sprite, size)
                self.sprites[name] = sprite
                logging.info(f"Successfully loaded sprite from {subdirectory_path}")
                return sprite
            except pygame.error as e:
                # If loading fails, try the next method
                logging.warning(f"Error loading from subdirectory: {e}")
        else:
            logging.debug(f"File not found at {subdirectory_path}")
        
        # Try to load the sprite from file directly in sprites_dir (assets/sprites/player.png)
        direct_path = os.path.join(self.sprites_dir, f"{name}.png")
        logging.debug(f"Checking direct path: {direct_path}")
        
        if os.path.exists(direct_path):
            logging.debug(f"Found sprite at {direct_path}")
            try:
                # Load and scale the sprite
                sprite = pygame.image.load(direct_path).convert_alpha()
                sprite = pygame.transform.scale(sprite, size)
                self.sprites[name] = sprite
                logging.info(f"Successfully loaded sprite from {direct_path}")
                return sprite
            except pygame.error as e:
                # If loading fails, fall back to placeholder
                logging.warning(f"Error loading from direct path: {e}")
        else:
            logging.debug(f"File not found at {direct_path}")

        # Create placeholder sprite
        logging.warning(f"Using placeholder for sprite '{name}' (no valid file found)")
        sprite = self._create_placeholder(name, size)
        self.sprites[name] = sprite
        return sprite

    def _create_placeholder(self, name: str, size: Tuple[int, int]) -> pygame.Surface:
        """
        Create a placeholder sprite with specified color.

        Args:
            name: Name of the entity (determines color)
            size: Size (width, height) of the sprite

        Returns:
            Placeholder sprite surface
        """
        # Get the placeholder color for the entity type
        color = self.placeholder_colors.get(name, (200, 200, 200))  # Default to gray

        # Create a surface with alpha channel
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        if name == "player":
            # Triangle shape for player
            width, height = size
            points = [
                (width // 2, 0),           # Top point
                (0, height),               # Bottom left
                (width, height)            # Bottom right
            ]
            pygame.draw.polygon(sprite, color, points)

        elif name == "enemy":
            # Pentagon shape for enemy
            width, height = size
            points = [
                (width // 2, 0),             # Top point
                (0, height // 2),            # Middle left
                (width // 4, height),        # Bottom left
                (3 * width // 4, height),    # Bottom right
                (width, height // 2)         # Middle right
            ]
            pygame.draw.polygon(sprite, color, points)

        elif name == "missile":
            # Elongated pentagon for missile
            width, height = size
            points = [
                (width // 2, 0),             # Top point
                (width // 4, height // 4),   # Upper left
                (0, height),                 # Bottom left
                (width, height),             # Bottom right
                (3 * width // 4, height // 4)  # Upper right
            ]
            pygame.draw.polygon(sprite, color, points)

        else:
            # Simple rectangle with slight transparency for other entities
            pygame.draw.rect(sprite, (*color, 220), pygame.Rect(0, 0, size[0], size[1]))

        return sprite

    def render(
        self,
        screen: pygame.Surface,
        entity_type: str,
        position: Union[Position, Tuple[float, float]],
        size: Tuple[int, int],
        rotation: float = 0
    ) -> None:
        """
        Render a sprite to the screen.

        Args:
            screen: Pygame surface to render to
            entity_type: Type of entity to render
            position: Position (x, y) coordinates
            size: Size (width, height) of the sprite
            rotation: Rotation angle in degrees
        """
        # Get position as (x, y) tuple
        if isinstance(position, dict):
            pos = (position["x"], position["y"])
        else:
            pos = position

        # Get the sprite surface
        sprite = self.load_sprite(entity_type, size)

        # Apply rotation if needed
        if rotation != 0:
            sprite = pygame.transform.rotate(sprite, rotation)

        # Blit to screen
        screen.blit(sprite, pos)

    def load_animation(
        self,
        name: str,
        size: Tuple[int, int],
        frames: int = 4
    ) -> List[pygame.Surface]:
        """
        Load or create a simple animation sequence.

        Args:
            name: Base name of the animation
            size: Size (width, height) of each frame
            frames: Number of frames in the animation

        Returns:
            List of animation frame surfaces
        """
        # Check if animation is already cached
        animation_key = f"{name}_{frames}"
        if animation_key in self.animations:
            logging.debug(f"Using cached animation for '{animation_key}'")
            return self.animations[animation_key]

        logging.info(f"Loading animation '{name}' with {frames} frames at size {size}")
        
        # Try to load animation frames from files
        animation_frames = []
        for i in range(frames):
            # Try subdirectory path first (assets/sprites/effects/explosion_0.png)
            subdirectory_path = os.path.join(self.sprites_dir, name, f"{name}_{i}.png")
            logging.debug(f"Checking frame {i} at subdirectory path: {subdirectory_path}")
            
            if os.path.exists(subdirectory_path):
                logging.debug(f"Found frame {i} at {subdirectory_path}")
                try:
                    # Load and scale the frame
                    frame = pygame.image.load(subdirectory_path).convert_alpha()
                    frame = pygame.transform.scale(frame, size)
                    animation_frames.append(frame)
                    logging.debug(f"Successfully loaded frame {i} from {subdirectory_path}")
                    continue
                except pygame.error as e:
                    logging.warning(f"Error loading frame {i} from subdirectory: {e}")
            else:
                logging.debug(f"Frame {i} not found at {subdirectory_path}")
            
            # Try direct path (assets/sprites/explosion_0.png)
            direct_path = os.path.join(self.sprites_dir, f"{name}_{i}.png")
            logging.debug(f"Checking frame {i} at direct path: {direct_path}")
            
            if os.path.exists(direct_path):
                logging.debug(f"Found frame {i} at {direct_path}")
                try:
                    # Load and scale the frame
                    frame = pygame.image.load(direct_path).convert_alpha()
                    frame = pygame.transform.scale(frame, size)
                    animation_frames.append(frame)
                    logging.debug(f"Successfully loaded frame {i} from {direct_path}")
                    continue
                except pygame.error as e:
                    # If loading fails, use placeholder
                    logging.warning(f"Error loading frame {i} from direct path: {e}")
                    animation_frames.append(self._create_placeholder(name, size))
                    continue
            else:
                logging.debug(f"Frame {i} not found at {direct_path}")
            
            # If no valid paths, create a placeholder
            logging.warning(f"Using placeholder for frame {i} of '{name}' (file not found)")
            animation_frames.append(self._create_animation_placeholder(name, size, i, frames))

        self.animations[animation_key] = animation_frames
        return animation_frames

    def _create_animation_placeholder(
        self,
        name: str,
        size: Tuple[int, int],
        frame_index: int,
        total_frames: int
    ) -> pygame.Surface:
        """
        Create a placeholder animation frame.

        Args:
            name: Name of the entity
            size: Size (width, height) of the sprite
            frame_index: Current frame index
            total_frames: Total number of frames

        Returns:
            Placeholder animation frame surface
        """
        # Get base sprite
        base_sprite = self._create_placeholder(name, size)

        # For animation, slightly modify the sprite based on frame index
        # This is a simple placeholder effect
        progress = frame_index / max(1, total_frames - 1)

        # Create a pulsing effect
        pulse_factor = 0.8 + 0.4 * abs((progress * 2) - 1)  # Values between 0.8 and 1.2

        # Apply the pulse effect
        width, height = size
        scaled_w = int(width * pulse_factor)
        scaled_h = int(height * pulse_factor)

        # Center the scaled sprite
        offset_x = (width - scaled_w) // 2
        offset_y = (height - scaled_h) // 2

        # Create a new surface for the animation frame
        frame = pygame.Surface(size, pygame.SRCALPHA)

        # Scale the base sprite and blit to the frame
        scaled_sprite = pygame.transform.scale(base_sprite, (scaled_w, scaled_h))
        frame.blit(scaled_sprite, (offset_x, offset_y))

        return frame
