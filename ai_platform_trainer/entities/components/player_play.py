import logging
import math
import os
import random
from typing import Dict, List, Optional

import pygame

from ai_platform_trainer.entities.components.missile import Missile


class PlayerPlay:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue (fallback)
        self.position = {"x": screen_width // 4, "y": screen_height // 2}
        self.step = 5
        self.missiles: List[Missile] = []
        
        # New features
        self.score = 0
        self.lives = 3
        self.missile_cooldown = 500  # milliseconds
        self.last_missile_time = 0
        self.max_missiles = 3  # Allow up to 3 missiles at once
        self.invulnerable = False
        self.invulnerable_time = 0
        self.invulnerable_duration = 2000  # 2 seconds
        
        # Load player sprite
        self.sprite = self._load_sprite()
        
        # Visual effects
        self.flash_effect = False
        self.flash_start = 0
        self.flash_duration = 200  # milliseconds

    def _load_sprite(self) -> pygame.Surface:
        """Load the player sprite from assets."""
        try:
            sprite_path = os.path.join("assets", "sprites", "player", "player.png")
            sprite = pygame.image.load(sprite_path)
            return pygame.transform.scale(sprite, (self.size, self.size))
        except (pygame.error, FileNotFoundError) as e:
            logging.error(f"Could not load player sprite: {e}")
            # Create a fallback surface if sprite loading fails
            fallback = pygame.Surface((self.size, self.size))
            fallback.fill(self.color)
            return fallback

    def reset(self) -> None:
        """Reset the player to initial state."""
        self.position = {"x": self.screen_width // 4, "y": self.screen_height // 2}
        self.missiles.clear()
        self.score = 0
        self.lives = 3
        self.invulnerable = False
        logging.info("Player has been reset to the initial position.")

    def handle_input(self) -> bool:
        """Process player input for movement. Returns False to exit the game."""
        keys = pygame.key.get_pressed()

        # WASD / Arrow key movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] -= self.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] += self.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] -= self.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] += self.step

        # Wrap-around logic
        if self.position["x"] < -self.size:
            self.position["x"] = self.screen_width
        elif self.position["x"] > self.screen_width:
            self.position["x"] = -self.size
        if self.position["y"] < -self.size:
            self.position["y"] = self.screen_height
        elif self.position["y"] > self.screen_height:
            self.position["y"] = -self.size

        return True

    def shoot_missile(self, enemy_pos: Optional[Dict[str, float]] = None) -> None:
        """Shoot a missile toward the enemy with improved mechanics."""
        current_time = pygame.time.get_ticks()
        
        # Check cooldown and missile limit
        if (current_time - self.last_missile_time < self.missile_cooldown or 
                len(self.missiles) >= self.max_missiles):
            return
            
        self.last_missile_time = current_time
        missile_start_x = self.position["x"] + self.size // 2
        missile_start_y = self.position["y"] + self.size // 2

        birth_time = current_time
        # Random lifespan from 0.75–2.25s to match training
        random_lifespan = random.randint(750, 2250)
        missile_speed = 6.0  # Slightly faster missiles

        # Determine initial velocity based on enemy position if available
        if enemy_pos is not None:
            # Calculate the angle toward the enemy's position
            angle = math.atan2(
                enemy_pos["y"] - missile_start_y, 
                enemy_pos["x"] - missile_start_x
            )
            # Add a small random deviation to simulate inaccuracy
            angle += random.uniform(-0.1, 0.1)  # deviation in radians
            vx = missile_speed * math.cos(angle)
            vy = missile_speed * math.sin(angle)
        else:
            vx = missile_speed
            vy = 0.0

        # Create a new missile object with calculated initial velocity and random lifespan
        missile = Missile(
            x=missile_start_x,
            y=missile_start_y,
            speed=missile_speed,
            vx=vx,
            vy=vy,
            birth_time=birth_time,
            lifespan=random_lifespan,
        )
        self.missiles.append(missile)
        logging.debug(
            f"Shot missile #{len(self.missiles)} - {len(self.missiles)}/{self.max_missiles} active"
        )

    def hit_enemy(self) -> None:
        """Handle scoring when a missile hits the enemy."""
        self.score += 100
        logging.info(f"Player scored a hit! Current score: {self.score}")

    def take_damage(self) -> None:
        """Handle player being hit by the enemy."""
        if self.invulnerable:
            return
            
        self.lives -= 1
        self.invulnerable = True
        self.invulnerable_time = pygame.time.get_ticks()
        self.flash_effect = True
        self.flash_start = pygame.time.get_ticks()
        
        logging.info(f"Player took damage! Lives remaining: {self.lives}")
        
        if self.lives <= 0:
            logging.info("Player out of lives!")

    def update(self, current_time: int) -> None:
        """Update player state, missiles, and effects."""
        # Update invulnerability
        if (self.invulnerable and 
                current_time - self.invulnerable_time >= self.invulnerable_duration):
            self.invulnerable = False
            
        # Update flash effect
        if self.flash_effect and current_time - self.flash_start >= self.flash_duration:
            self.flash_effect = False
            
        # Update missiles
        self.update_missiles()

    def update_missiles(self) -> None:
        """Update all active missiles."""
        current_time = pygame.time.get_ticks()
        for missile in self.missiles[:]:
            missile.update()

            # Remove if it expires or goes off-screen
            if current_time - missile.birth_time >= missile.lifespan:
                self.missiles.remove(missile)
                continue

            if (
                missile.pos["x"] < 0
                or missile.pos["x"] > self.screen_width
                or missile.pos["y"] < 0
                or missile.pos["y"] > self.screen_height
            ):
                self.missiles.remove(missile)

    def draw_missiles(self, screen: pygame.Surface) -> None:
        """Draw all active missiles."""
        for missile in self.missiles:
            missile.draw(screen)

    def draw_ui(self, screen: pygame.Surface) -> None:
        """Draw score and lives UI."""
        font = pygame.font.Font(None, 36)
        # Score
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = font.render(f"Lives: {self.lives}", True, (255, 255, 255))
        screen.blit(lives_text, (self.screen_width - 120, 10))
        
        # Missiles
        missile_text = font.render(
            f"Missiles: {len(self.missiles)}/{self.max_missiles}", 
            True, 
            (255, 255, 255)
        )
        screen.blit(missile_text, (10, 50))

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the player, missiles, and UI elements."""
        # Draw player sprite or rectangle with invulnerability flashing
        if self.invulnerable and (pygame.time.get_ticks() // 200) % 2 == 0:
            # Flash effect by skipping draw every other 200ms
            pass
        else:
            screen.blit(self.sprite, (self.position["x"], self.position["y"]))
        
        # Draw missiles
        self.draw_missiles(screen)
        
        # Draw UI elements
        self.draw_ui(screen)

    def get_rect(self) -> pygame.Rect:
        """Get the player's collision rectangle."""
        return pygame.Rect(
            self.position["x"], 
            self.position["y"], 
            self.size, 
            self.size
        )
