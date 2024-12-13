import pygame
import random
import math

class Player:
    def __init__(self, screen_width, screen_height):
        self.position = {"x": screen_width // 2, "y": screen_height // 2}
        self.size = 50
        self.color = (0, 102, 204)
        self.step = 5
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Speed multiplier for trying to evade
        self.speed_multiplier = 1.2

        # Number of directions to sample each frame
        self.num_directions = 8

        # Keep track of how many times we fail to find good open space
        self.stall_counter = 0
        self.stall_threshold = 50  # After 50 stalls, try more random angles

        # Slight random factor to break predictability
        self.random_factor = 0.1

    def reset(self):
        self.position = {
            "x": self.screen_width // 2,
            "y": self.screen_height // 2
        }
        self.stall_counter = 0

    def update(self, enemy_x, enemy_y):
        # We'll sample multiple directions and pick the best
        # Directions are evenly spaced around the player
        best_score = -float('inf')
        best_dx, best_dy = 0, 0

        # Compute player's current distance from enemy
        px, py = self.position["x"], self.position["y"]
        dx_enemy = px - enemy_x
        dy_enemy = py - enemy_y
        dist_enemy = math.sqrt(dx_enemy*dx_enemy + dy_enemy*dy_enemy) + 0.0001

        run_away_speed = self.step * self.speed_multiplier

        # Generate candidate angles
        for i in range(self.num_directions):
            angle = (2 * math.pi / self.num_directions) * i
            # Add a small random tweak to angle to avoid predictability
            angle += random.uniform(-0.2, 0.2) * (1 + self.stall_counter / (self.stall_threshold*2))

            dx = math.cos(angle)
            dy = math.sin(angle)

            future_x = px + dx * run_away_speed
            future_y = py + dy * run_away_speed

            # Check bounds and measure open space
            open_space_score = self.measure_open_space(px, py, dx, dy, run_away_speed)

            # Distance from enemy if move in this direction
            ex = future_x - enemy_x
            ey = future_y - enemy_y
            future_dist = math.sqrt(ex*ex + ey*ey) + 0.0001

            # Score for this direction:
            # - Prefer greater future distance from enemy
            # - Prefer more open space
            # Add slight randomness to break ties
            score = (future_dist / dist_enemy) * 1.5 + open_space_score + random.uniform(-self.random_factor, self.random_factor)

            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

        # If no improvement in space, increment stall counter
        if best_score < 0.5:  # Arbitrary threshold, adjust as needed
            self.stall_counter += 1
        else:
            self.stall_counter = max(0, self.stall_counter - 1)

        # Move player in chosen direction
        future_x = px + best_dx * run_away_speed
        future_y = py + best_dy * run_away_speed

        # Clamp final position
        self.position["x"] = max(0, min(self.screen_width - self.size, future_x))
        self.position["y"] = max(0, min(self.screen_height - self.size, future_y))

    def measure_open_space(self, start_x, start_y, dx, dy, step_size):
        # Cast a ray forward until hitting a wall (or max distance) to measure open space
        max_check_dist = self.screen_width  # Large enough to test across arena
        increments = int(max_check_dist / step_size)
        score = 0.0

        cur_x, cur_y = start_x, start_y
        for _ in range(increments):
            cur_x += dx * step_size
            cur_y += dy * step_size
            if (cur_x < 0 or cur_x > self.screen_width - self.size or
                cur_y < 0 or cur_y > self.screen_height - self.size):
                # Hit a wall
                break
            score += 0.1  # Each free step adds to the score
        return score

    def get_position(self):
        return self.position

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.position["x"], self.position["y"], self.size, self.size))