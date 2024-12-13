import pygame
import random
import math


class Enemy:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (173, 153, 228)
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}

        self.base_speed = max(2, screen_width // 400)
        self.state_timer = 0
        self.current_pattern = None

        self.patterns = ["random_walk", "circle_move", "diagonal_move"]

        # Initialize forced escape variables BEFORE calling switch_pattern()
        self.wall_stall_counter = 0
        self.wall_stall_threshold = 10
        self.forced_escape_timer = 0
        self.forced_angle = None
        self.forced_speed = None

        # Pattern-specific variables
        self.circle_center = (self.pos["x"], self.pos["y"])
        self.circle_angle = 0.0
        self.circle_radius = 100
        self.diagonal_direction = (1, 1)

        # Now safe to call switch_pattern(), since forced_escape_timer exists
        self.switch_pattern()

    def switch_pattern(self):
        if self.forced_escape_timer > 0:
            return

        new_pattern = self.current_pattern
        while new_pattern == self.current_pattern:
            new_pattern = random.choice(self.patterns)
        self.current_pattern = new_pattern
        self.state_timer = random.randint(120, 300)

        if self.current_pattern == "circle_move":
            self.circle_center = (self.pos["x"], self.pos["y"])
            self.circle_angle = random.uniform(0, 2 * math.pi)
            self.circle_radius = random.randint(50, 150)
        elif self.current_pattern == "diagonal_move":
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            self.diagonal_direction = (dx, dy)

    def update_movement(self, player_x, player_y, player_speed):
        if self.forced_escape_timer > 0:
            self.forced_escape_timer -= 1
            self.apply_forced_escape_movement()
        else:
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

        # Check wall hugging
        if self.is_hugging_wall():
            self.wall_stall_counter += 1
        else:
            self.wall_stall_counter = max(0, self.wall_stall_counter - 1)

        # If hugging wall too long, forced escape
        if (
            self.wall_stall_counter > self.wall_stall_threshold
            and self.forced_escape_timer <= 0
        ):
            self.initiate_forced_escape()

    def initiate_forced_escape(self):
        dist_left = self.pos["x"]
        dist_right = (self.screen_width - self.size) - self.pos["x"]
        dist_top = self.pos["y"]
        dist_bottom = (self.screen_height - self.size) - self.pos["y"]

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_dist == dist_left:
            base_angle = 0.0
        elif min_dist == dist_right:
            base_angle = math.pi
        elif min_dist == dist_top:
            base_angle = math.pi / 2
        else:
            base_angle = math.pi * 3 / 2

        min_angle_variation = math.radians(30)
        angle_choice = base_angle + random.uniform(
            -min_angle_variation, min_angle_variation
        )
        self.forced_angle = angle_choice
        self.forced_speed = self.base_speed * random.uniform(1.5, 2.5)
        self.forced_escape_timer = random.randint(1, 30)  # ~0.5 sec if 60FPS

        self.wall_stall_counter = 0
        self.state_timer = self.forced_escape_timer * 2

    def apply_forced_escape_movement(self):
        dx = math.cos(self.forced_angle) * self.forced_speed
        dy = math.sin(self.forced_angle) * self.forced_speed
        self.pos["x"] += dx
        self.pos["y"] += dy

    def is_hugging_wall(self):
        # Consider hugging wall if within 20px of any wall
        wall_margin = 20
        if (
            self.pos["x"] < wall_margin
            or self.pos["x"] > self.screen_width - self.size - wall_margin
            or self.pos["y"] < wall_margin
            or self.pos["y"] > self.screen_height - self.size - wall_margin
        ):
            return True
        return False

    def random_walk_pattern(self):
        if not hasattr(self, "random_walk_timer") or self.random_walk_timer <= 0:
            self.random_walk_angle = random.uniform(0, 2 * math.pi)
            self.random_walk_speed = self.base_speed * random.uniform(0.5, 2.0)
            self.random_walk_timer = random.randint(30, 90)
        else:
            self.random_walk_timer -= 1

        dx = math.cos(self.random_walk_angle) * self.random_walk_speed
        dy = math.sin(self.random_walk_angle) * self.random_walk_speed
        self.pos["x"] += dx
        self.pos["y"] += dy

    def circle_pattern(self):
        speed = self.base_speed * 1.5
        angle_increment = 0.02 * (speed / self.base_speed)
        self.circle_angle += angle_increment

        dx = math.cos(self.circle_angle) * self.circle_radius
        dy = math.sin(self.circle_angle) * self.circle_radius
        self.pos["x"] = self.circle_center[0] + dx
        self.pos["y"] = self.circle_center[1] + dy

        if random.random() < 0.01:
            self.circle_radius += random.randint(-5, 5)
            self.circle_radius = max(20, min(200, self.circle_radius))

    def diagonal_pattern(self):
        if random.random() < 0.05:
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
