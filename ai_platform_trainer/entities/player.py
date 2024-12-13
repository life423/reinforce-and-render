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
        # Calculate vector from enemy to player (player_x - enemy_x, player_y - enemy_y)
        dx = self.position["x"] - enemy_x
        dy = self.position["y"] - enemy_y

        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            dx /= dist
            dy /= dist
        else:
            # If dist == 0, just pick a random direction to move
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length

        # Apply a small random turn to the direction
        # Convert (dx, dy) to an angle
        angle = math.atan2(dy, dx)
        # Add a small random angle turn
        angle += random.uniform(-self.turn_angle_range_rad, self.turn_angle_range_rad)
        # Convert back to vector
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Move the player at a slightly faster speed than base step
        run_away_speed = self.step * self.speed_multiplier
        self.position["x"] += dx * run_away_speed
        self.position["y"] += dy * run_away_speed

        # Clamp player position to screen bounds
        self.position["x"] = max(0, min(self.screen_width - self.size, self.position["x"]))
        self.position["y"] = max(0, min(self.screen_height - self.size, self.position["y"]))

    def get_position(self):
        return self.position

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.position["x"], self.position["y"], self.size, self.size))