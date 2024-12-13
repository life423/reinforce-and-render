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

        # Probability of ignoring the enemy and moving randomly
        # Higher means more collisions and variation
        self.random_move_chance = 0.2

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
        # Decide whether to move away from the enemy or move randomly
        if random.random() < self.random_move_chance:
            # Move in a random direction
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle)
            dy = math.sin(angle)
        else:
            # Move away from the enemy
            dx = self.position["x"] - enemy_x
            dy = self.position["y"] - enemy_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.0001:
                # If too close or same spot, pick a random escape angle
                angle = random.uniform(0, 2 * math.pi)
                dx, dy = math.cos(angle), math.sin(angle)
            else:
                # Normalize direction away from enemy
                dx /= dist
                dy /= dist

            # Add slight randomness so it's not a perfect line
            angle = math.atan2(dy, dx)
            angle += random.uniform(-0.2, 0.2)  # small random turn
            dx, dy = math.cos(angle), math.sin(angle)

        # Move the player
        self.position["x"] += dx * self.step
        self.position["y"] += dy * self.step

        # Clamp position to avoid going out of bounds
        self.clamp_position()

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
