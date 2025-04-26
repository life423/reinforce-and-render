"""
Enemy entity for AI Platform Trainer.

This module defines the Enemy class that uses both supervised and reinforcement
learning to challenge the player.
"""
import logging
import math
import os
import random
from typing import List, Tuple

import numpy as np
import pygame

from ai_platform_trainer.entities.behaviors.enemy_ai_controller import update_enemy_movement

# Import optionally - will be None if not available
try:
    from stable_baselines3 import PPO as SB3PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    SB3PPO = None
    logging.info("Stable Baselines 3 not available - RL features disabled")


class EnemyPlay:
    def __init__(self, screen_width: int, screen_height: int, model=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        # Size is interpreted as width and height
        self.width = self.size
        self.height = self.size
        self.color = (255, 215, 0)  # Gold (fallback color)
        
        # Position is the CENTER of the enemy, not top-left corner
        self.position = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        # Also maintain pos for compatibility with any code that uses it
        self.pos = self.position
        self.model = model
        self.base_speed = max(2, screen_width // 400)
        self.visible = True
        self.angle = 0  # Rotation angle

        # Fade-in attributes
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_duration = 300  # milliseconds
        self.fade_start_time = 0
        
        # Load sprite
        self.sprite = self._load_sprite()
        self.using_sprite = self.sprite is not None
        
        # Visual effects
        self.pulse_effect = True
        self.pulse_min = 0.9
        self.pulse_max = 1.1
        self.pulse_speed = 0.005
        self.pulse_value = 1.0
        self.pulse_direction = 1
        
        # Particle effects
        self.particles: List[dict] = []
        self.max_particles = 15
        self.particle_colors = [(255, 140, 0), (255, 165, 0), (255, 215, 0)]  # Fire colors
        
        # Create fallback Surface if sprite isn't available
        if not self.using_sprite:
            self.surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            self.surface.fill((*self.color, 255))

    def _load_sprite(self) -> pygame.Surface:
        """Load the enemy sprite from assets."""
        try:
            sprite_path = os.path.join("assets", "sprites", "enemy", "enemy.png")
            sprite = pygame.image.load(sprite_path)
            return pygame.transform.scale(sprite, (self.size, self.size))
        except (pygame.error, FileNotFoundError) as e:
            logging.error(f"Could not load enemy sprite: {e}")
            return None

    def wrap_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply wrap-around logic to keep enemy on screen.
        """
        def wrap(val: float, lower: float, upper: float) -> float:
            if val < lower:
                return upper
            elif val > upper:
                return lower
            return val

        new_x = wrap(x, -self.size, self.screen_width)
        new_y = wrap(y, -self.size, self.screen_height)
        return new_x, new_y

    def load_rl_model(self, model_path: str = "models/enemy_rl/final_model.zip") -> bool:
        """
        Load a trained reinforcement learning model for enemy behavior.

        Args:
            model_path: Path to the saved PPO model

        Returns:
            True if successful, False otherwise
        """
        if not RL_AVAILABLE or SB3PPO is None:
            logging.warning("Cannot load RL model - Stable Baselines 3 not available")
            return False

        try:
            self.rl_model = SB3PPO.load(model_path)
            self.using_rl = True
            logging.info(f"Loaded RL model from {model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load RL model: {e}")
            self.using_rl = False
            return False

    def apply_rl_action(self, action: np.ndarray) -> None:
        """
        Apply an action from the reinforcement learning model.

        Args:
            action: Normalized action vector from the RL model (-1 to 1 range)
        """
        if not self.visible:
            return

        # Store previous position for angle calculation
        prev_x, prev_y = self.position["x"], self.position["y"]
        
        # Scale action to actual speed
        speed = self.base_speed
        self.position["x"] += action[0] * speed
        self.position["y"] += action[1] * speed

        # Apply wrap-around
        # Break up long line for position wrap
        new_x, new_y = self.wrap_position(
            self.position["x"], 
            self.position["y"]
        )
        self.position["x"], self.position["y"] = new_x, new_y
        
        # Update rotation angle based on movement direction
        x_changed = abs(self.pos["x"] - prev_x) > 0.01
        y_changed = abs(self.pos["y"] - prev_y) > 0.01
        if x_changed or y_changed:
            self.angle = math.degrees(math.atan2(
                self.pos["y"] - prev_y, 
                self.pos["x"] - prev_x
            ))
            
        # Add particles based on movement
        self._add_movement_particles()

    def update_movement(
        self,
        player_x: float,
        player_y: float,
        player_speed: int,
        current_time: int
    ):
        """
        Update enemy movement based on AI model or reinforcement learning.
        """
        if not self.visible:
            return
            
        # Store previous position for angle calculation
        prev_x, prev_y = self.pos["x"], self.pos["y"]

        # Use RL model if available
        if hasattr(self, 'using_rl') and self.using_rl and hasattr(self, 'rl_model'):
            # Create observation for the model
            screen_width, screen_height = self.screen_width, self.screen_height

            # Normalize values to help with training stability
            px = player_x / screen_width
            py = player_y / screen_height
            ex = self.pos["x"] / screen_width
            ey = self.pos["y"] / screen_height

            # Calculate distance between player and enemy centers
            dx = player_x - self.pos["x"]
            dy = player_y - self.pos["y"]
            dist = math.sqrt(dx**2 + dy**2) / max(screen_width, screen_height)

            player_speed_norm = player_speed / 10.0
            time_factor = 0.5  # Placeholder for time since last hit

            # Create observation array
            obs = np.array([
                px, py, ex, ey, dist, player_speed_norm, time_factor
            ], dtype=np.float32)

            # Get action from model
            try:
                action, _ = self.rl_model.predict(obs, deterministic=False)
                self.apply_rl_action(action)
            except Exception as e:
                # Fallback to traditional approach on error
                logging.error(
                    f"RL model inference failed: {e}. Falling back to neural network."
                )
                self.using_rl = False
                update_enemy_movement(self, player_x, player_y, player_speed, current_time)
        else:
            # Use traditional neural network approach
            update_enemy_movement(
                self,
                player_x=player_x,
                player_y=player_y,
                player_speed=player_speed,
                current_time=current_time,
            )
            
        # Calculate angle to face the player if we moved
        if abs(self.pos["x"] - prev_x) > 0.01 or abs(self.pos["y"] - prev_y) > 0.01:
            # Update angle to face movement direction
            self.angle = math.degrees(math.atan2(
                self.pos["y"] - prev_y, 
                self.pos["x"] - prev_x
            ))
            
            # Add particles based on movement
            self._add_movement_particles()
        
        # Update visual effects
        self._update_visual_effects(current_time)

    def _add_movement_particles(self) -> None:
        """Add particle effects based on enemy movement."""
        if len(self.particles) >= self.max_particles:
            return
            
        # Add particle at enemy's center position
        if random.random() < 0.3:  # Only add particles sometimes
            self.particles.append({
                'x': self.position["x"],
                'y': self.position["y"],
                'size': random.randint(2, 5),
                'color': random.choice(self.particle_colors),
                'alpha': 255,
                'decay': random.uniform(3, 7),
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(-0.5, 0.5)
            })

    def _update_visual_effects(self, current_time: int) -> None:
        """Update all visual effects."""
        # Update particles
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['alpha'] -= particle['decay']
            
            if particle['alpha'] <= 0:
                self.particles.remove(particle)
                
        # Update pulse effect
        if self.pulse_effect:
            self.pulse_value += self.pulse_direction * self.pulse_speed
            if self.pulse_value > self.pulse_max:
                self.pulse_value = self.pulse_max
                self.pulse_direction = -1
            elif self.pulse_value < self.pulse_min:
                self.pulse_value = self.pulse_min
                self.pulse_direction = 1
                
        # Update fade-in effect
        self.update_fade_in(current_time)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the enemy and its visual effects."""
        if not self.visible:
            return
            
        # Draw particles
        for particle in self.particles:
            size = particle['size']
            particle_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                particle_surface,
                (*particle['color'], int(particle['alpha'])),
                (particle['size'], particle['size']),
                particle['size']
            )
            screen.blit(
                particle_surface, 
                (particle['x'] - size, particle['y'] - size)
            )
        
        # Draw enemy
        if self.using_sprite:
            # Create scaled copy for pulse effect
            scaled_size = int(self.size * self.pulse_value)
            scaled_sprite = pygame.transform.scale(self.sprite, (scaled_size, scaled_size))
            
            # Rotate to face movement direction
            rotated_sprite = pygame.transform.rotate(scaled_sprite, -self.angle)
            
            # Get rect for centered blitting
            rect = rotated_sprite.get_rect(center=(
                int(self.position["x"]),
                int(self.position["y"])
            ))
            
            # Draw debug visualization if enabled
            debug_mode = getattr(self, 'show_debug', False)
            if debug_mode:
                # Draw center point
                pygame.draw.circle(
                    screen,
                    (255, 0, 0),  # Red for center
                    (int(self.position["x"]), int(self.position["y"])),
                    3
                )
                
                # Draw bounding box
                pygame.draw.rect(
                    screen,
                    (0, 255, 0),  # Green for hitbox
                    self.get_rect(),
                    1  # Line width
                )
            
            # Apply fade alpha
            rotated_sprite.set_alpha(self.fade_alpha)
            
            # Draw the enemy
            screen.blit(rotated_sprite, rect)
        else:
            # Fallback to simple surface if sprite isn't available
            self.surface.set_alpha(self.fade_alpha)
            # Calculate top-left corner for surface blitting
            top_left_x = self.position["x"] - self.width // 2
            top_left_y = self.position["y"] - self.height // 2
            screen.blit(self.surface, (top_left_x, top_left_y))

    def hide(self) -> None:
        """Hide the enemy after being hit."""
        self.visible = False
        logging.info("Enemy hidden due to collision.")

    def show(self, current_time: int) -> None:
        """Show the enemy with fade-in effect."""
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time
        self.particles = []  # Clear any existing particles
        logging.info("Enemy set to fade in.")

    def update_fade_in(self, current_time: int) -> None:
        """Update fade-in effect progress."""
        if self.fading_in:
            elapsed = current_time - self.fade_start_time
            if elapsed >= self.fade_duration:
                self.fade_alpha = 255
                self.fading_in = False
                logging.info("Enemy fade-in completed.")
            else:
                self.fade_alpha = int((elapsed / self.fade_duration) * 255)

    def set_position(self, x: int, y: int) -> None:
        """Set the enemy's position."""
        self.pos["x"], self.pos["y"] = x, y
        
    def get_rect(self) -> pygame.Rect:
        """Get the enemy's collision rectangle."""
        # Return a rectangle centered on enemy's position
        return pygame.Rect(
            self.position["x"] - self.width // 2,
            self.position["y"] - self.height // 2, 
            self.width, 
            self.height
        )
