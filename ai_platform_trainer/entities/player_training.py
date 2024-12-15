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
        # Increase minimum from 120 to 180 for less frequent switches
        self.state_timer = random.randint(180, 300)

        if self.current_pattern == "circle_move":
            # Clamp circle_center inside screen boundaries to reduce sudden off-screen jumps
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

    def random_walk_pattern(self):
        if self.random_walk_timer <= 0:
            self.random_walk_angle = random.uniform(0, 2 * math.pi)
            # Lower upper bound from 2.0 to 1.2 to reduce speed extremes
            self.random_walk_speed = self.step * random.uniform(0.5, 1.2)
            self.random_walk_timer = random.randint(30, 90)
        else:
            self.random_walk_timer -= 1

        dx = math.cos(self.random_walk_angle) * self.random_walk_speed
        dy = math.sin(self.random_walk_angle) * self.random_walk_speed
        self.position["x"] += dx
        self.position["y"] += dy

    def circle_pattern(self):
        speed = self.step
        angle_increment = 0.02 * (speed / self.step)
        self.circle_angle += angle_increment

        dx = math.cos(self.circle_angle) * self.circle_radius
        dy = math.sin(self.circle_angle) * self.circle_radius
        self.position["x"] = self.circle_center[0] + dx
        self.position["y"] = self.circle_center[1] + dy

        if random.random() < 0.01:
            self.circle_radius += random.randint(-5, 5)
            self.circle_radius = max(20, min(200, self.circle_radius))

    def diagonal_pattern(self):
        # Reduce angle change frequency from 0.05 to 0.02 for smoother movement
        if random.random() < 0.02:
            angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
            angle += random.uniform(-0.3, 0.3)
            self.diagonal_direction = (math.cos(angle), math.sin(angle))

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

        if dist < close_threshold:
            # Enemy close: use random_walk to move away randomly
            self.random_walk_pattern()
        elif dist > far_threshold:
            # Enemy far: pick pattern
            if self.current_pattern == "circle_move":
                self.circle_pattern()
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern()
            else:
                self.random_walk_pattern()
        else:
            # Neutral zone: follow current pattern
            if self.current_pattern == "random_walk":
                self.random_walk_pattern()
            elif self.current_pattern == "circle_move":
                self.circle_pattern()
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern()

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
