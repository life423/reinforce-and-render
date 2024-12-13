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

        # Timing logic
        self.frame_counter = 0
        self.direction_timer = (
            0  # Counts down how long to keep moving in current direction
        )
        self.current_dx = 0
        self.current_dy = 0

    def reset(self):
        """Reset player position and movement state."""
        self.position = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.frame_counter = 0
        self.direction_timer = 0
        self.current_dx = 0
        self.current_dy = 0

    def update(self, enemy_x, enemy_y):
        """Update the player's position with wrap-around logic."""
        self.frame_counter += 1

        # Direction logic
        if self.direction_timer <= 0 and self.frame_counter % 20 == 0:
            # Pick a random direction
            direction = random.choice(["up", "down", "left", "right"])
            if direction == "up":
                self.current_dx, self.current_dy = 0, -1
            elif direction == "down":
                self.current_dx, self.current_dy = 0, 1
            elif direction == "left":
                self.current_dx, self.current_dy = -1, 0
            elif direction == "right":
                self.current_dx, self.current_dy = 1, 0
            self.direction_timer = 90  # Move in this direction for 90 frames

        if self.direction_timer > 0:
            # Update position based on current direction
            self.position["x"] += self.current_dx * self.step
            self.position["y"] += self.current_dy * self.step
            self.direction_timer -= 1

        # Apply wrap-around logic
        self.position["x"], self.position["y"] = self.wrap_position(
            self.position["x"],
            self.position["y"],
            self.screen_width,
            self.screen_height,
            self.size,
        )

    @staticmethod
    def wrap_position(x, y, width, height, size):
        """Wrap around the screen if the player moves off the edges."""
        if x < -size:
            x = width
        elif x > width:
            x = -size
        if y < -size:
            y = height
        elif y > height:
            y = -size
        return x, y

    def draw(self, screen):
        """Draw the player on the screen."""
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
