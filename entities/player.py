import pygame  # <-- Add this line to import pygame
import random
from noise import pnoise1


class Player:
    def __init__(self, screen_width, screen_height):
        self.position = {"x": screen_width // 2, "y": screen_height // 2}
        self.size = 50
        self.color = (0, 102, 204)
        self.step = 5
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Noise parameters for smooth movement
        self.noise_offset_x = random.uniform(
            0, 100)  # Random noise offset for x-axis
        self.noise_offset_y = random.uniform(
            0, 100)  # Random noise offset for y-axis
        self.noise_time = 0.0  # Time variable for noise-based movement

    def reset(self):
        # Reset player position to the center of the screen
        self.position = {"x": self.screen_width //
                         2, "y": self.screen_height // 2}
        self.noise_time = 0.0  # Reset noise time

    def update_noise_movement(self):
        # Increment time for smoother movement
        self.noise_time += 0.01

        # Generate new positions using Perlin noise
        dx = pnoise1(self.noise_time + self.noise_offset_x) * self.step
        dy = pnoise1(self.noise_time + self.noise_offset_y) * self.step

        # Update position while keeping it within screen bounds
        self.position["x"] = max(
            0, min(self.screen_width - self.size, self.position["x"] + dx))
        self.position["y"] = max(
            0, min(self.screen_height - self.size, self.position["y"] + dy))

    def draw(self, screen):
        # Use pygame to draw the player rectangle
        pygame.draw.rect(
            screen, self.color, (self.position["x"], self.position["y"], self.size, self.size))
