import pygame
import random
import math


class Enemy:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (255, 0, 0)  # Red for enemy
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}

        self.base_speed = max(2, screen_width // 400)
        self.state_timer = 0
        self.current_pattern = None

        # Patterns: random_walk, circle_move, diagonal_move
        self.patterns = ["random_walk", "circle_move", "diagonal_move"]
        self.switch_pattern()  # pick initial pattern

        # For circle pattern
        self.circle_center = (self.pos["x"], self.pos["y"])
        self.circle_angle = 0.0
        self.circle_radius = 100

        # For diagonal pattern
        self.diagonal_direction = (1, 1)

    def switch_pattern(self):
        # Pick a new pattern different from the current
        new_pattern = self.current_pattern
        while new_pattern == self.current_pattern:
            new_pattern = random.choice(self.patterns)
        self.current_pattern = new_pattern

        # Reset timer: each pattern lasts 2 to 5 seconds of movement
        self.state_timer = random.randint(120, 300)  # frames (~2-5 secs at 60FPS)

        if self.current_pattern == "circle_move":
            # Recenter circle around current pos
            self.circle_center = (self.pos["x"], self.pos["y"])
            self.circle_angle = random.uniform(0, 2 * math.pi)
            self.circle_radius = random.randint(50, 150)

        elif self.current_pattern == "diagonal_move":
            # Pick a random diagonal direction
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            self.diagonal_direction = (dx, dy)

    def update_movement(self, player_x, player_y, player_speed):
        # We're not using player_x, player_y, player_speed for logic here
        # The enemy just moves in its pattern to generate diverse data.

        # Decrement timer, switch pattern if time’s up
        self.state_timer -= 1
        if self.state_timer <= 0:
            self.switch_pattern()

        if self.current_pattern == "random_walk":
            self.random_walk_pattern()
        elif self.current_pattern == "circle_move":
            self.circle_pattern()
        elif self.current_pattern == "diagonal_move":
            self.diagonal_pattern()

        # Clamp position
        self.pos["x"] = max(0, min(self.screen_width - self.size, self.pos["x"]))
        self.pos["y"] = max(0, min(self.screen_height - self.size, self.pos["y"]))

    def random_walk_pattern(self):
        # Enemy picks a random direction each frame with slight speed variance
        angle = random.uniform(0, 2 * math.pi)
        speed = self.base_speed * random.uniform(0.5, 2.0)
        dx = math.cos(angle) * speed
        dy = math.sin(angle) * speed
        self.pos["x"] += dx
        self.pos["y"] += dy

    def circle_pattern(self):
        # Enemy moves in a circle around a fixed center
        # Gradually increase the angle to move along the circle
        speed = self.base_speed * 1.5
        self.circle_angle += 0.02  # Increase angle slowly
        dx = math.cos(self.circle_angle) * self.circle_radius
        dy = math.sin(self.circle_angle) * self.circle_radius
        self.pos["x"] = self.circle_center[0] + dx
        self.pos["y"] = self.circle_center[1] + dy

        # Occasionally vary radius for some irregularity
        if random.random() < 0.01:
            self.circle_radius += random.randint(-5, 5)
            self.circle_radius = max(20, min(200, self.circle_radius))

    def diagonal_pattern(self):
        # Enemy moves steadily in a diagonal direction
        # Occasionally change direction slightly
        if random.random() < 0.05:
            # small angle tweak
            angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
            angle += random.uniform(-0.3, 0.3)
            self.diagonal_direction = (math.cos(angle), math.sin(angle))

        speed = self.base_speed * random.uniform(1.0, 2.0)
        self.pos["x"] += self.diagonal_direction[0] * speed
        self.pos["y"] += self.diagonal_direction[1] * speed

    def draw(self, screen):
        pygame.draw.rect(
            screen, self.color, (self.pos["x"], self.pos["y"], self.size, self.size)
        )
