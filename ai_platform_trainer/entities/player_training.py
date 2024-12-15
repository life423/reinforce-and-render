import random
import math
import logging
import pygame
from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.utils.helpers import wrap_position


class PlayerTraining:
    PATTERNS = ["random_walk", "circle_move", "diagonal_move"]

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.step = 5
        self.missiles = []
        logging.info("PlayerTraining initialized.")

        self.desired_distance = 200
        self.margin = 20

        # Pattern-related attributes
        self.current_pattern = None
        self.state_timer = 0
        self.random_walk_timer = 0
        self.random_walk_angle = 0.0
        # Fixed speed to reduce jitter and complexity
        self.random_walk_speed = self.step
        self.circle_angle = 0.0
        self.circle_radius = 100
        self.diagonal_direction = (1, 1)

        self.switch_pattern()

    def switch_pattern(self):
        new_pattern = self.current_pattern
        while new_pattern == self.current_pattern:
            new_pattern = random.choice(self.PATTERNS)

        self.current_pattern = new_pattern
        # Longer stable periods before switching patterns to reduce jitter
        self.state_timer = random.randint(180, 300)

        if self.current_pattern == "circle_move":
            # Clamp circle_center to prevent large off-screen jumps
            cx = max(self.size, min(self.screen_width - self.size, self.position["x"]))
            cy = max(self.size, min(self.screen_height - self.size, self.position["y"]))
            self.circle_center = (cx, cy)
            self.circle_angle = random.uniform(0, 2 * math.pi)
            self.circle_radius = random.randint(50, 150)
        elif self.current_pattern == "diagonal_move":
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            self.diagonal_direction = (dx, dy)

        logging.debug(f"Switched pattern to {self.current_pattern} at {self.position}")

    def reset(self) -> None:
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.missiles.clear()
        self.switch_pattern()
        logging.info("PlayerTraining has been reset.")

    def bias_angle_away_from_enemy(self, enemy_x, enemy_y, base_angle):
        # Compute enemy angle relative to player
        dx = enemy_x - self.position["x"]
        dy = enemy_y - self.position["y"]
        dist = math.hypot(dx, dy)
        if dist == 0:
            # If overlapping enemy, choose opposite direction
            return (base_angle + math.pi) % (2 * math.pi)

        enemy_angle = math.atan2(dy, dx)

        close_threshold = self.desired_distance - self.margin
        far_threshold = self.desired_distance + self.margin

        # Determine bias strength: closer enemy means stronger bias away
        if dist < close_threshold:
            # Very close: strongly bias (~90 degrees)
            bias_strength = math.radians(90)
        elif dist > far_threshold:
            # Far: small bias (~15 degrees)
            bias_strength = math.radians(15)
        else:
            # Medium distance: moderate bias (~45 degrees)
            bias_strength = math.radians(45)

        # Determine which side to rotate based on angle difference
        angle_diff = (base_angle - enemy_angle) % (2 * math.pi)

        # If angle_diff < pi, player currently angled somewhat towards enemy; rotate by +bias
        # else rotate by -bias to push away.
        if angle_diff < math.pi:
            new_angle = base_angle + bias_strength
        else:
            new_angle = base_angle - bias_strength

        return new_angle % (2 * math.pi)

    def random_walk_pattern(self, enemy_x, enemy_y):
        if self.random_walk_timer <= 0:
            self.random_walk_angle = random.uniform(0, 2 * math.pi)
            # Fixed speed = self.step, no changes
            self.random_walk_speed = self.step
            self.random_walk_timer = random.randint(30, 90)
        else:
            self.random_walk_timer -= 1

        # Bias angle away from enemy
        angle = self.bias_angle_away_from_enemy(
            enemy_x, enemy_y, self.random_walk_angle
        )

        dx = math.cos(angle) * self.random_walk_speed
        dy = math.sin(angle) * self.random_walk_speed
        self.position["x"] += dx
        self.position["y"] += dy

    def circle_pattern(self, enemy_x, enemy_y):
        speed = self.step
        angle_increment = 0.02 * (speed / self.step)
        self.circle_angle += angle_increment

        # Before applying circle coords, consider biasing circle_angle slightly
        # We'll treat circle_angle as base and adjust it slightly:
        # Convert circle_angle to vector, add slight angle away from enemy
        # Actually simpler: circle_angle defines a point on circle. Just find final angle after bias.
        # We'll pick a small intermediate angle offset based on bias:
        # Instead of rewriting circle logic extensively, just do a tiny final angle tweak after
        # calculating dx, dy. But that would distort the circle. Instead, nudge circle_angle itself.

        # We'll bias the final movement direction:
        # final_angle = bias_angle_away_from_enemy will require a direction vector.
        # Circle direction is tangent. Let's just pick a direction from center to player pos:
        # Actually, circle pattern sets exact coords based on circle_center and circle_angle.
        # Let's first find final coords, then we have direction vector from circle_center:
        dx = math.cos(self.circle_angle) * self.circle_radius
        dy = math.sin(self.circle_angle) * self.circle_radius

        # The direction we are about to move is from circle_center to these coords:
        # Convert that to angle:
        base_angle = math.atan2(dy, dx)
        final_angle = self.bias_angle_away_from_enemy(enemy_x, enemy_y, base_angle)

        # After bias, recalculate dx, dy from final_angle but same radius:
        dx = math.cos(final_angle) * self.circle_radius
        dy = math.sin(final_angle) * self.circle_radius

        self.position["x"] = self.circle_center[0] + dx
        self.position["y"] = self.circle_center[1] + dy

        # Smaller circle radius adjustments to reduce sudden jumps
        if random.random() < 0.01:
            self.circle_radius += random.randint(-2, 2)
            self.circle_radius = max(20, min(200, self.circle_radius))

    def diagonal_pattern(self, enemy_x, enemy_y):
        if random.random() < 0.02:
            angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
            angle += random.uniform(-0.3, 0.3)
            self.diagonal_direction = (math.cos(angle), math.sin(angle))

        # Convert diagonal direction to angle, then bias it
        base_angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
        final_angle = self.bias_angle_away_from_enemy(enemy_x, enemy_y, base_angle)

        # Update diagonal_direction based on final_angle to keep it consistent
        self.diagonal_direction = (math.cos(final_angle), math.sin(final_angle))

        speed = self.step
        self.position["x"] += self.diagonal_direction[0] * speed
        self.position["y"] += self.diagonal_direction[1] * speed

    def update(self, enemy_x: float, enemy_y: float) -> None:
        dist = math.hypot(self.position["x"] - enemy_x, self.position["y"] - enemy_y)
        close_threshold = self.desired_distance - self.margin
        far_threshold = self.desired_distance + self.margin

        self.state_timer -= 1
        if self.state_timer <= 0:
            self.switch_pattern()

        # Distance-based pattern selection:
        if dist < close_threshold:
            # Enemy close: Use random_walk but biased away from enemy
            self.random_walk_pattern(enemy_x, enemy_y)
        elif dist > far_threshold:
            # Enemy far: use chosen pattern but always angle away from enemy
            if self.current_pattern == "circle_move":
                self.circle_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern(enemy_x, enemy_y)
            else:
                self.random_walk_pattern(enemy_x, enemy_y)
        else:
            # Neutral zone: follow current pattern with angle bias away from enemy
            if self.current_pattern == "random_walk":
                self.random_walk_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "circle_move":
                self.circle_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern(enemy_x, enemy_y)

        # Wrap-around logic
        self.position["x"], self.position["y"] = wrap_position(
            self.position["x"],
            self.position["y"],
            self.screen_width,
            self.screen_height,
            self.size,
        )

    def shoot_missile(self) -> None:
        if len(self.missiles) == 0:
            missile_start_x = self.position["x"] + self.size // 2
            missile_start_y = self.position["y"] + self.size // 2
            missile = Missile(x=missile_start_x, y=missile_start_y, vx=5.0, vy=0.0)
            self.missiles.append(missile)
            logging.info("Training mode: Missile shot straight to the right.")
        else:
            logging.debug(
                "Attempted to shoot a missile in training mode, but one is already active."
            )

    def update_missiles(self) -> None:
        for missile in self.missiles[:]:
            missile.update()
            if (
                missile.pos["x"] < 0
                or missile.pos["x"] > self.screen_width
                or missile.pos["y"] < 0
                or missile.pos["y"] > self.screen_height
            ):
                self.missiles.remove(missile)
                logging.debug("Missile removed for going off-screen.")

    def draw_missiles(self, screen: pygame.Surface) -> None:
        for missile in self.missiles:
            missile.draw(screen)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
        self.draw_missiles(screen)
