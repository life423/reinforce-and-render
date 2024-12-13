import pygame
import math


class Enemy:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        # Pastel Lavender color for enemy
        self.color = (173, 153, 228)
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.speed = max(2, screen_width // 400)

    def update_movement(self, player_x, player_y, player_speed):
        # Move directly toward the player
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            dx /= dist
            dy /= dist
        # Move enemy at a speed somewhat related to player_speed or fixed
        enemy_speed = player_speed * 0.7
        self.pos["x"] += dx * enemy_speed
        self.pos["y"] += dy * enemy_speed

        # Clamp position to prevent going out of bounds
        self.pos["x"] = max(0, min(self.screen_width - self.size, self.pos["x"]))
        self.pos["y"] = max(0, min(self.screen_height - self.size, self.pos["y"]))

    def draw(self, screen):
        pygame.draw.rect(
            screen, self.color, (self.pos["x"], self.pos["y"], self.size, self.size)
        )
