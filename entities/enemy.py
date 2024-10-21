import pygame  # Import pygame
import random
from noise import pnoise1


class Enemy:
    def __init__(self, screen_width, screen_height):
        # Initial position in the center
        self.start_x = screen_width // 2
        self.start_y = screen_height // 2
        self.pos = {"x": self.start_x, "y": self.start_y}

        # Set size to match the player's size
        self.size = 50  # Updated to be consistent with the player block size
        # Orange color to match the color palette (or similar)
        self.color = (255, 165, 0)

        self.speed = max(2, screen_width // 400)  # Speed for dynamic movement
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Noise parameters for smooth movement
        self.noise_offset_x = random.uniform(
            0, 100)  # Random noise offset for x-axis
        self.noise_offset_y = random.uniform(
            0, 100)  # Random noise offset for y-axis
        self.time = 0.0  # Time variable for noise-based movement

        # Random movement direction change timer
        self.direction_change_timer = random.randint(30, 100)

        # Current velocity
        self.velocity_x = 0
        self.velocity_y = 0

    def reset(self):
        # Reset the enemy to its starting position and reset noise time
        self.pos = {"x": self.start_x, "y": self.start_y}
        self.time = 0.0
        self.velocity_x = 0
        self.velocity_y = 0

    def update(self):
        # Increment time to get new positions from noise
        self.time += 0.01  # Increment time for smoother changes

        # Generate noise-based movement for smooth base movement
        noise_dx = pnoise1(self.time + self.noise_offset_x) * self.speed
        noise_dy = pnoise1(self.time + self.noise_offset_y) * self.speed

        # Randomly change movement direction every few frames
        self.direction_change_timer -= 1
        if self.direction_change_timer <= 0:
            self.velocity_x = random.choice([-1, 1]) * self.speed
            self.velocity_y = random.choice([-1, 1]) * self.speed
            # Reset the timer for the next direction change
            self.direction_change_timer = random.randint(30, 100)

        # Combine noise-based movement with random directional movement
        self.pos["x"] += noise_dx + self.velocity_x
        self.pos["y"] += noise_dy + self.velocity_y

        # Keep the position within bounds
        self.pos["x"] = max(
            0, min(self.screen_width - self.size, self.pos["x"]))
        self.pos["y"] = max(
            0, min(self.screen_height - self.size, self.pos["y"]))

    def draw(self, screen):
        # Use pygame to draw the enemy rectangle
        pygame.draw.rect(screen, self.color,
                         (self.pos["x"], self.pos["y"], self.size, self.size))
