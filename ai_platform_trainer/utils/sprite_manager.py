"""
Sprite manager for AI Platform Trainer.

This module handles loading, caching, and rendering sprites for game entities.
It provides an abstraction layer over pygame's sprite handling to make the
code more modular and maintainable.
"""
import glob
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
            "missile": (255, 255, 50),  # Yellow
            "wall": (100, 100, 100),    # Gray
            "rock": (150, 120, 90),     # Brown
            "obstacle": (120, 120, 120)  # Dark Gray
        }
        
        # Mapping of entity types to their sprite subdirectories
        self.entity_type_dirs = {
            "player": "player",
            "enemy": "enemy", 
            "missile": "missile",
            "wall": "obstacles",
            "rock": "obstacles",
            "obstacle": "obstacles",
            "explosion": "effects"
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
        # Use hash of name+size tuple as cache key
        cache_key = f"{name}_{size[0]}x{size[1]}"
        if cache_key in self.sprites:
            logging.debug(f"Using cached sprite for '{name}'")
            return self.sprites[cache_key]

        logging.info(f"Loading sprite '{name}' with size {size}")
        
        # Determine potential paths with clear priority
        potential_paths = self._get_potential_sprite_paths(name)
        
        # Try each path
        for path in potential_paths:
            if os.path.exists(path):
                try:
                    # Load and scale the sprite
                    sprite = pygame.image.load(path).convert_alpha()
                    sprite = pygame.transform.scale(sprite, size)
                    self.sprites[cache_key] = sprite
                    logging.info(f"Successfully loaded sprite from {path}")
                    return sprite
                except pygame.error as e:
                    # Try next path if this one fails
                    logging.warning(f"Error loading from {path}: {e}")
        
        # Create placeholder sprite if no paths worked
        logging.warning(f"Using placeholder for sprite '{name}' (no valid file found)")
        sprite = self._create_placeholder(name, size)
        self.sprites[cache_key] = sprite
        return sprite
        
    def _get_potential_sprite_paths(self, name: str) -> List[str]:
        """Get prioritized list of potential sprite paths."""
        paths = []
        
        # Handle base sprite names (for category lookup)
        base_name = name
        if "_" in name and name.split("_")[-1].isdigit():
            # Handle animation frames like "explosion_0"
            base_name = name.split("_")[0]
        
        # 1. Category subfolder with full name if this entity type has a mapping
        if base_name in self.entity_type_dirs:
            category_dir = self.entity_type_dirs[base_name]
            paths.append(os.path.join(self.sprites_dir, category_dir, f"{name}.png"))
        
        # 2. Direct in sprites directory (legacy support)
        paths.append(os.path.join(self.sprites_dir, f"{name}.png"))
        
        # 3. Generic subfolder with name
        paths.append(os.path.join(self.sprites_dir, name, f"{name}.png"))
        
        logging.debug(f"Potential paths for sprite '{name}': {paths}")
        return paths

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
            
        elif name in ["wall", "rock", "obstacle"]:
            # For obstacles, add a distinctive border to clearly mark placeholders
            pygame.draw.rect(sprite, color, pygame.Rect(0, 0, size[0], size[1]))
            pygame.draw.rect(sprite, (255, 0, 255), pygame.Rect(0, 0, size[0], size[1]), width=2)

        else:
            # Simple rectangle with slight transparency for other entities
            pygame.draw.rect(sprite, (*color, 220), pygame.Rect(0, 0, size[0], size[1]))
            
        # Add a debug marker to easily identify placeholders
        font_size = max(10, min(size) // 4)
        try:
            font = pygame.font.SysFont('Arial', font_size)
            text = font.render('PH', True, (255, 255, 255))
            text_rect = text.get_rect(center=(size[0]//2, size[1]//2))
            sprite.blit(text, text_rect)
        except Exception:
            # If font rendering fails, add a cross instead
            pygame.draw.line(sprite, (255, 255, 255), (0, 0), size, 2)
            pygame.draw.line(sprite, (255, 255, 255), (0, size[1]), (size[0], 0), 2)

        return sprite
        
    def is_placeholder(self, sprite: pygame.Surface) -> bool:
        """
        Check if a sprite is a placeholder.
        
        Args:
            sprite: The sprite to check
            
        Returns:
            True if the sprite is a placeholder, False otherwise
        """
        # Sample some pixels to check for placeholder patterns
        if not sprite:
            return True
            
        width, height = sprite.get_size()
        
        # Check if the sprite has our placeholder markers
        try:
            # Check for the white "PH" text in the center (our placeholder marker)
            center_pixel = sprite.get_at((width//2, height//2))
            
            # If center pixel is white, it's likely our placeholder text
            is_ph = (center_pixel[0] > 240 and center_pixel[1] > 240 and center_pixel[2] > 240)
            
            return is_ph
        except IndexError:
            return False

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
        base_name: str,
        size: Tuple[int, int],
        frames: int = None
    ) -> List[pygame.Surface]:
        """
        Load or create a simple animation sequence with glob pattern matching.

        Args:
            base_name: Base name of the animation
            size: Size (width, height) of each frame
            frames: Maximum number of frames to load (optional)

        Returns:
            List of animation frame surfaces
        """
        # Cache key includes base name and size
        cache_key = f"anim_{base_name}_{size[0]}x{size[1]}"
        if cache_key in self.animations:
            logging.debug(f"Using cached animation for '{cache_key}'")
            return self.animations[cache_key]

        logging.info(f"Loading animation '{base_name}' with size {size}")
        
        # Determine potential animation frame patterns
        patterns = []
        
        # Try category folder first if it exists
        if base_name in self.entity_type_dirs:
            category_dir = self.entity_type_dirs[base_name]
            pattern = os.path.join(self.sprites_dir, category_dir, f"{base_name}_*.png")
            patterns.append(pattern)
        
        # Try direct pattern in sprites directory
        patterns.append(os.path.join(self.sprites_dir, f"{base_name}_*.png"))
        
        # Find matching files
        frame_files = []
        for pattern in patterns:
            matches = sorted(glob.glob(pattern))
            if matches:
                frame_files = matches
                logging.debug(f"Found {len(matches)} frames with pattern {pattern}")
                break
        
        # Limit to requested frame count if specified
        if frames is not None and len(frame_files) > frames:
            frame_files = frame_files[:frames]
        
        # If no frames found and frames count specified, use placeholders
        if not frame_files and frames is not None:
            logging.warning(f"No animation frames found for '{base_name}', creating {frames} placeholders")
            animation_frames = [
                self._create_animation_placeholder(base_name, size, i, frames)
                for i in range(frames)
            ]
            self.animations[cache_key] = animation_frames
            return animation_frames
            
        # Load each frame
        animation_frames = []
        for i, file_path in enumerate(frame_files):
            try:
                frame = pygame.image.load(file_path).convert_alpha()
                frame = pygame.transform.scale(frame, size)
                animation_frames.append(frame)
                logging.debug(f"Loaded animation frame from {file_path}")
            except pygame.error as e:
                logging.warning(f"Error loading animation frame from {file_path}: {e}")
                ph_frame = self._create_animation_placeholder(
                    base_name, size, i, len(frame_files)
                )
                animation_frames.append(ph_frame)
        
        self.animations[cache_key] = animation_frames
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
        
        # Add frame indicator text
        font_size = max(10, min(size) // 5)
        try:
            font = pygame.font.SysFont('Arial', font_size)
            frame_text = f'{frame_index+1}/{total_frames}'
            text = font.render(frame_text, True, (255, 255, 255))
            # Position the text below center
            x_pos = size[0] // 2
            y_pos = size[1] // 2 + size[1] // 3
            center_pos = (x_pos, y_pos)
            text_rect = text.get_rect(center=center_pos)
            frame.blit(text, text_rect)
        except Exception:
            pass

        return frame
        
    def validate_sprites(self, expected_sprites=None):
        """
        Validate that all required sprites can be loaded correctly.
        
        Args:
            expected_sprites: List of sprite names to validate
            
        Returns:
            List of sprites that could not be loaded properly
        """
        if expected_sprites is None:
            # Default list of critical sprites
            expected_sprites = [
                "player", "enemy", "missile", 
                "wall", "rock", "obstacle",
                "explosion_0", "explosion_1", "explosion_2", "explosion_3"
            ]
            
        # Standard size for testing
        test_size = (50, 50)
        
        results = {}
        failures = []
        
        for name in expected_sprites:
            try:
                sprite = self.load_sprite(name, test_size)
                is_ph = self.is_placeholder(sprite)
                status = "OK" if not is_ph else "PLACEHOLDER"
                results[name] = {"status": status, "is_placeholder": is_ph}
                
                if is_ph:
                    failures.append(name)
            except Exception as e:
                results[name] = {"status": "ERROR", "error": str(e)}
                failures.append(name)
        
        # Log results
        for name, result in results.items():
            logging.info(f"Sprite '{name}': {result['status']}")
            
        return failures
