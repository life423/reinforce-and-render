import pygame
import random
import math

class Player:
    def __init__(self, screen_width, screen_height):
        self.position = {"x": screen_width // 2, "y": screen_height // 2}
        self.size = 50  # Size of the player block
        self.color = (0, 102, 204)  # A shade of blue
        self.step = 5
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Speed multiplier for running away from the enemy
        self.speed_multiplier = 1.2

        # Small random turn angle range (in degrees)
        self.turn_angle_range = 10  # +/- 10 degrees
        # Convert degrees to radians for math functions
        self.turn_angle_range_rad = math.radians(self.turn_angle_range)

    def reset(self):
        # Reset player position to the center of the screen
        self.position = {
            "x": self.screen_width // 2,
            "y": self.screen_height // 2
        }

    def update(self, enemy_x, enemy_y):
        dx = self.position["x"] - enemy_x
        dy = self.position["y"] - enemy_y
        dist = (dx*dx + dy*dy)**0.5

        if dist > 0:
            dx /= dist
            dy /= dist
        else:
            # If player is exactly on enemy, pick a random direction
            angle = random.uniform(0, 2*math.pi)
            dx, dy = math.cos(angle), math.sin(angle)

        # Apply a small random turn
        angle = math.atan2(dy, dx)
        angle += random.uniform(-self.turn_angle_range_rad, self.turn_angle_range_rad)
        dx, dy = math.cos(angle), math.sin(angle)

        run_away_speed = self.step * self.speed_multiplier

        # Check if going straight will hit a wall
        future_x = self.position["x"] + dx * run_away_speed
        future_y = self.position["y"] + dy * run_away_speed

        # If future position is out of bounds, adjust angle
        attempts = 10
        while attempts > 0 and (future_x < 0 or future_x > self.screen_width - self.size or future_y < 0 or future_y > self.screen_height - self.size):
            # Try a slightly different angle that avoids the wall
            angle += random.uniform(-0.3, 0.3)  # adjust as needed
            dx, dy = math.cos(angle), math.sin(angle)
            future_x = self.position["x"] + dx * run_away_speed
            future_y = self.position["y"] + dy * run_away_speed
            attempts -= 1

        # Move player
        self.position["x"] = max(0, min(self.screen_width - self.size, future_x))
        self.position["y"] = max(0, min(self.screen_height - self.size, future_y))

    def get_position(self):
        return self.position

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.position["x"], self.position["y"], self.size, self.size))