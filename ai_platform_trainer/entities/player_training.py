import pygame
import random
import math


class Player:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        # Pastel Green for player
        self.color = (112, 191, 113)
        self.step = 5
        self.position = {"x": self.screen_width // 2, "y": self.screen_height // 2}

    def reset(self):
        self.position = {"x": self.screen_width // 2, "y": self.screen_height // 2}

    def clamp_position(self):
        self.position["x"] = max(
            0, min(self.position["x"], self.screen_width - self.size)
        )
        self.position["y"] = max(
            0, min(self.position["y"], self.screen_height - self.size)
        )

    def update(self, enemy_x, enemy_y):
        # Calculate vector from enemy to player (we want to move along this vector to increase distance)
        dx = self.position["x"] - enemy_x
        dy = self.position["y"] - enemy_y
        dist = math.sqrt(dx * dx + dy * dy)

        # If enemy is on same spot or too close, pick a random direction to escape
        if dist < 0.0001:
            angle = random.uniform(0, 2 * math.pi)
            dx, dy = math.cos(angle), math.sin(angle)
        else:
            # Normalize (dx, dy) to length 1
            dx /= dist
            dy /= dist

        # Add slight random variation to not be too predictable
        angle = math.atan2(dy, dx)
        angle += random.uniform(-0.3, 0.3)  # random turn of up to ~17 degrees
        dx, dy = math.cos(angle), math.sin(angle)

        # Move player away from the enemy
        self.position["x"] += dx * self.step
        self.position["y"] += dy * self.step

        # Clamp to ensure we don't go out of bounds
        self.clamp_position()

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
