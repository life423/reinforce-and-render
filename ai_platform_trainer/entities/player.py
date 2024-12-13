import pygame
import random
import math
from collections import deque

class Player:
    def __init__(self, screen_width, screen_height):
        self.position = {"x": screen_width // 2, "y": screen_height // 2}
        self.size = 50
        self.color = (0, 102, 204)
        self.step = 5
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Speed multiplier base
        self.speed_multiplier = 1.2
        self.num_directions = 8

        # Track stalls and repeated patterns
        self.stall_counter = 0
        self.stall_threshold = 50
        self.random_factor = 0.1

        # Memory of last chosen directions
        self.direction_memory = deque(maxlen=5)

        # Thresholds for desperate escape
        self.desperate_dist_threshold = 100.0  # if enemy too close
        self.desperate_stall_threshold = 20     # if stuck too many frames with enemy close

        self.close_counter = 0  # counts how many consecutive frames enemy is too close

    def reset(self):
        self.position = {
            "x": self.screen_width // 2,
            "y": self.screen_height // 2
        }
        self.stall_counter = 0
        self.close_counter = 0
        self.direction_memory.clear()

    def update(self, enemy_x, enemy_y, enemy_dx=0, enemy_dy=0):
        # enemy_dx, enemy_dy: estimate of enemy direction (if available), else pass 0,0
        # This helps the player anticipate enemy movement a bit.
        
        px, py = self.position["x"], self.position["y"]
        dx_enemy = px - enemy_x
        dy_enemy = py - enemy_y
        dist_enemy = math.sqrt(dx_enemy*dx_enemy + dy_enemy*dy_enemy) + 0.0001

        run_away_speed = self.step * self.speed_multiplier

        # Check if we are too close to the enemy
        if dist_enemy < self.desperate_dist_threshold:
            self.close_counter += 1
        else:
            self.close_counter = max(0, self.close_counter - 1)

        # If player is stuck too close to enemy for too long, try a desperate escape
        desperate_mode = (self.close_counter > self.desperate_stall_threshold)

        # Speed variation sometimes
        if random.random() < 0.05:
            run_away_speed *= random.uniform(0.9, 1.1)

        best_score = -float('inf')
        best_dx, best_dy = 0, 0

        # Sample directions
        for i in range(self.num_directions):
            angle = (2 * math.pi / self.num_directions) * i
            # Add randomness to angle, more if stalled
            angle += random.uniform(-0.2, 0.2) * (1 + self.stall_counter / (self.stall_threshold*2))

            dx = math.cos(angle)
            dy = math.sin(angle)

            future_x = px + dx * run_away_speed
            future_y = py + dy * run_away_speed

            # Score direction
            score = self.score_direction(px, py, dx, dy, run_away_speed,
                                         enemy_x, enemy_y, dist_enemy,
                                         enemy_dx, enemy_dy, desperate_mode)

            # Penalize repeating similar directions
            for old_dx, old_dy in self.direction_memory:
                similarity = dx*old_dx + dy*old_dy  # dot product for direction similarity
                if similarity > 0.9:  # very similar direction
                    score -= 0.2

            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

        # Update stall counter
        if best_score < 0.5:
            self.stall_counter += 1
        else:
            self.stall_counter = max(0, self.stall_counter - 1)

        # If desperate mode and best_score still low, try a big random jump perpendicular to enemy
        if desperate_mode and best_score < 0.5:
            # Choose a direction roughly perpendicular to enemy vector
            angle_perp = math.atan2(dy_enemy, dx_enemy) + math.pi/2 + random.uniform(-0.5,0.5)
            ddx = math.cos(angle_perp)
            ddy = math.sin(angle_perp)
            run_away_speed *= 2.0  # boost speed for a desperate leap
            future_x = px + ddx * run_away_speed
            future_y = py + ddy * run_away_speed
            # Clamp
            future_x = max(0, min(self.screen_width - self.size, future_x))
            future_y = max(0, min(self.screen_height - self.size, future_y))
            self.position["x"], self.position["y"] = future_x, future_y
            # Reset some counters after a desperate move
            self.stall_counter = 0
            self.close_counter = 0
            self.direction_memory.append((ddx, ddy))
            return

        # Move player in chosen direction
        future_x = px + best_dx * run_away_speed
        future_y = py + best_dy * run_away_speed

        # Clamp final position
        self.position["x"] = max(0, min(self.screen_width - self.size, future_x))
        self.position["y"] = max(0, min(self.screen_height - self.size, future_y))

        # Store chosen direction
        self.direction_memory.append((best_dx, best_dy))

    def score_direction(self, px, py, dx, dy, speed, enemy_x, enemy_y, dist_enemy, enemy_dx, enemy_dy, desperate_mode):
        # Evaluate direction based on multiple factors

        future_x = px + dx * speed
        future_y = py + dy * speed

        # Distance to enemy after move
        ex = future_x - enemy_x
        ey = future_y - enemy_y
        future_dist = math.sqrt(ex*ex + ey*ey) + 0.0001

        dist_ratio = future_dist / dist_enemy

        # Open space
        open_space_score = self.measure_open_space(px, py, dx, dy, speed)

        # Wall proximity: penalize being too close to wall after move
        wall_dist_score = self.wall_avoidance_score(future_x, future_y)

        # Predict enemy next pos (simple: enemy_x+enemy_dx, enemy_y+enemy_dy)
        # Avoid directions where future enemy might be closer
        predicted_enemy_x = enemy_x + enemy_dx * 10  # predict a step ahead
        predicted_enemy_y = enemy_y + enemy_dy * 10
        ep_x = future_x - predicted_enemy_x
        ep_y = future_y - predicted_enemy_y
        predicted_dist = math.sqrt(ep_x*ep_x + ep_y*ep_y) + 0.0001
        enemy_predict_score = (predicted_dist / future_dist) * 0.5  # small weight

        # Slight randomness
        rand_bonus = random.uniform(-self.random_factor, self.random_factor)

        score = dist_ratio * 1.5 + open_space_score + wall_dist_score + enemy_predict_score + rand_bonus

        # If in desperate mode, emphasize increasing enemy distance even more
        if desperate_mode:
            score += dist_ratio  # extra emphasis on getting away

        return score

    def measure_open_space(self, start_x, start_y, dx, dy, step_size):
        # Cast a ray forward until hitting a wall or max distance
        max_check_dist = self.screen_width
        increments = int(max_check_dist / step_size)
        score = 0.0
        cur_x, cur_y = start_x, start_y
        for _ in range(increments):
            cur_x += dx * step_size
            cur_y += dy * step_size
            if (cur_x < 0 or cur_x > self.screen_width - self.size or
                cur_y < 0 or cur_y > self.screen_height - self.size):
                break
            score += 0.1
        return score

    def wall_avoidance_score(self, x, y):
        # Instead of pushing player to center, just avoid being too close to walls
        # Closer to wall = lower score
        dist_left = x
        dist_right = (self.screen_width - self.size) - x
        dist_top = y
        dist_bottom = (self.screen_height - self.size) - y

        # Minimum distance to any wall
        min_dist_wall = min(dist_left, dist_right, dist_top, dist_bottom)
        # More distance from wall = better
        # Scale this so that closer than 50 px to a wall reduces score
        wall_avoid_weight = 0.5
        if min_dist_wall < 50:
            # penalize heavily if too close
            return (min_dist_wall / 50.0) * wall_avoid_weight - wall_avoid_weight
        else:
            # If decently away from walls, slight positive score
            return wall_avoid_weight * 0.1

    def get_position(self):
        return self.position

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.position["x"], self.position["y"], self.size, self.size))