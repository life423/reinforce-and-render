import random
import pygame
from noise import pnoise1


class Enemy:
    def __init__(self, screen_width, screen_height):
        self.start_x = screen_width // 2
        self.start_y = screen_height // 2
        self.pos = {"x": self.start_x, "y": self.start_y}
        self.size = 100
        self.color = (255, 99, 71)  # Adjusted color to fit the palette
        self.speed = max(1, screen_width // 500)
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Noise parameters
        self.noise_offset_x = random.uniform(0, 100)
        self.noise_offset_y = random.uniform(0, 100)
        self.time = 0.0

    def reset(self):
        # Reset the enemy to its starting position
        self.pos = {"x": self.start_x, "y": self.start_y}

    def update_noise_movement(self):
        # Increment time to get a new position
        self.time += 0.01  # Increment time for smoother changes

        # Generate new positions using Perlin noise
        dx = pnoise1(self.time + self.noise_offset_x) * self.speed
        dy = pnoise1(self.time + self.noise_offset_y) * self.speed

        # Update position while keeping it within bounds
        self.pos["x"] = max(
            0, min(self.screen_width - self.size, self.pos["x"] + dx))
        self.pos["y"] = max(
            0, min(self.screen_height - self.size, self.pos["y"] + dy))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color,
                         (self.pos["x"], self.pos["y"], self.size, self.size))
