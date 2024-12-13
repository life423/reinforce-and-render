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
        self.position = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.frame_counter = 0
        self.direction_timer = 0
        self.current_dx = 0
        self.current_dy = 0

    def clamp_position(self):
        # self.position["x"] = max(
        #     0, min(self.position["x"], self.screen_width - self.size)
        # )
        # self.position["y"] = max(
        #     0, min(self.position["y"], self.screen_height - self.size)
        # )
        if self.position["x"] < -self.size:
            self.position["x"] = self.screen_width
        elif self.position["x"] > self.screen_width:
            self.position["x"] = -self.size

        if self.position["y"] < -self.size:
            self.position["y"] = self.screen_height
        elif self.position["y"] > self.screen_height:
            self.position["y"] = -self.size

    def update(self, enemy_x, enemy_y):
        self.frame_counter += 1

        # If we are not currently locked into a direction (direction_timer <=0)
        # and the current frame is a multiple of 20, pick a new direction
        if self.direction_timer <= 0 and self.frame_counter % 20 == 0:
            # Pick a new direction from four options: up, down, left, right
            direction = random.choice(["up", "down", "left", "right"])
            if direction == "up":
                self.current_dx, self.current_dy = 0, -1
            elif direction == "down":
                self.current_dx, self.current_dy = 0, 1
            elif direction == "left":
                self.current_dx, self.current_dy = -1, 0
            elif direction == "right":
                self.current_dx, self.current_dy = 1, 0

            # Set direction_timer for 1.5 seconds at 60 FPS = 90 frames
            self.direction_timer = 90

        if self.direction_timer > 0:
            # Move in the chosen direction
            self.position["x"] += self.current_dx * self.step
            self.position["y"] += self.current_dy * self.step
            self.direction_timer -= 1
        else:
            # Not currently locked into a direction, do nothing
            pass

        # Clamp position to avoid going out of bounds
        self.clamp_position()

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
