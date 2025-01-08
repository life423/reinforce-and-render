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

        self.current_pattern = None
        self.state_timer = 0
        self.random_walk_timer = 0
        self.random_walk_angle = 0.0
        self.random_walk_speed = self.step
        self.circle_angle = 0.0
        self.circle_radius = 100
        self.diagonal_direction = (1, 1)

        # Wrap-around cooldown attributes
        self.wrap_cooldown = 0
        self.wrap_cooldown_frames = 120

        # Introduce a velocity vector for smoother movement (to reduce jitter)
        self.velocity = {"x": 0.0, "y": 0.0}
        self.velocity_blend_factor = (
            0.2  # Lower = smoother movement, Higher = more responsive
        )

        self.switch_pattern()

    def switch_pattern(self):
        new_pattern = self.current_pattern
        while new_pattern == self.current_pattern:
            new_pattern = random.choice(self.PATTERNS)

        self.current_pattern = new_pattern
        self.state_timer = random.randint(180, 300)

        if self.current_pattern == "circle_move":
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
        dx = enemy_x - self.position["x"]
        dy = enemy_y - self.position["y"]
        dist = math.hypot(dx, dy)
        if dist == 0:
            return (base_angle + math.pi) % (2 * math.pi)

        enemy_angle = math.atan2(dy, dx)
        if dist < self.desired_distance - self.margin:
            bias_strength = math.radians(30)
        elif dist > self.desired_distance + self.margin:
            bias_strength = math.radians(15)
        else:
            bias_strength = math.radians(45)

        angle_diff = (base_angle - enemy_angle) % (2 * math.pi)
        if angle_diff < math.pi:
            new_angle = base_angle + bias_strength
        else:
            new_angle = base_angle - bias_strength

        return new_angle % (2 * math.pi)

    def move_with_velocity(self, ndx, ndy):
        """
        Use a velocity vector to reduce jitter.
        Blend current velocity toward (ndx * step, ndy * step).
        """
        target_vx = ndx * self.step
        target_vy = ndy * self.step

        self.velocity["x"] = (1 - self.velocity_blend_factor) * self.velocity[
            "x"
        ] + self.velocity_blend_factor * target_vx
        self.velocity["y"] = (1 - self.velocity_blend_factor) * self.velocity[
            "y"
        ] + self.velocity_blend_factor * target_vy

        self.position["x"] += self.velocity["x"]
        self.position["y"] += self.velocity["y"]

    def random_walk_pattern(self, enemy_x, enemy_y):
        if self.random_walk_timer <= 0:
            self.random_walk_angle = random.uniform(0, 2 * math.pi)
            self.random_walk_speed = self.step
            self.random_walk_timer = random.randint(30, 90)
        else:
            self.random_walk_timer -= 1

        angle = self.bias_angle_away_from_enemy(
            enemy_x, enemy_y, self.random_walk_angle
        )
        ndx = math.cos(angle)
        ndy = math.sin(angle)
        self.move_with_velocity(ndx, ndy)

        logging.debug(f"Random walk: pos={self.position}")

    def circle_pattern(self, enemy_x, enemy_y):
        angle_increment = 0.02
        self.circle_angle += angle_increment

        desired_x = (
            self.circle_center[0] + math.cos(self.circle_angle) * self.circle_radius
        )
        desired_y = (
            self.circle_center[1] + math.sin(self.circle_angle) * self.circle_radius
        )

        dx = desired_x - self.position["x"]
        dy = desired_y - self.position["y"]

        base_angle = math.atan2(dy, dx)
        final_angle = self.bias_angle_away_from_enemy(enemy_x, enemy_y, base_angle)
        ndx = math.cos(final_angle)
        ndy = math.sin(final_angle)
        self.move_with_velocity(ndx, ndy)

        if random.random() < 0.01:
            self.circle_radius += random.randint(-2, 2)
            self.circle_radius = max(20, min(200, self.circle_radius))

        logging.debug(
            f"Circle move: pos={self.position}, center={self.circle_center}, radius={self.circle_radius}"
        )

    def diagonal_pattern(self, enemy_x, enemy_y):
        if random.random() < 0.02:
            angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
            angle += random.uniform(-0.3, 0.3)
            self.diagonal_direction = (math.cos(angle), math.sin(angle))

        base_angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
        final_angle = self.bias_angle_away_from_enemy(enemy_x, enemy_y, base_angle)

        self.diagonal_direction = (math.cos(final_angle), math.sin(final_angle))
        ndx, ndy = self.diagonal_direction
        self.move_with_velocity(ndx, ndy)

        logging.debug(
            f"Diagonal move: pos={self.position}, direction={self.diagonal_direction}"
        )

    def update(self, enemy_x: float, enemy_y: float) -> None:
        dist = math.hypot(self.position["x"] - enemy_x, self.position["y"] - enemy_y)
        close_threshold = self.desired_distance - self.margin
        far_threshold = self.desired_distance + self.margin

        self.state_timer -= 1
        if self.state_timer <= 0:
            self.switch_pattern()

        # Distance-based pattern selection
        if dist < close_threshold:
            self.random_walk_pattern(enemy_x, enemy_y)
        elif dist > far_threshold:
            if self.current_pattern == "circle_move":
                self.circle_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern(enemy_x, enemy_y)
            else:
                self.random_walk_pattern(enemy_x, enemy_y)
        else:
            if self.current_pattern == "random_walk":
                self.random_walk_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "circle_move":
                self.circle_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern(enemy_x, enemy_y)

        # Introduce wrap-around cooldown logic
        old_x, old_y = self.position["x"], self.position["y"]

        if self.wrap_cooldown == 0:
            new_x, new_y = wrap_position(
                self.position["x"],
                self.position["y"],
                self.screen_width,
                self.screen_height,
                self.size,
            )

            if (new_x, new_y) != (old_x, old_y):
                self.wrap_cooldown = self.wrap_cooldown_frames
                self.position["x"], self.position["y"] = new_x, new_y
                logging.debug(f"Wrapped around: new pos={self.position}")
            else:
                self.position["x"], self.position["y"] = new_x, new_y
        else:
            # During cooldown, adjust movement to prevent moving off-screen
            self.position["x"] = max(
                0,
                min(self.position["x"], self.screen_width - self.size)
            )
            self.position["y"] = max(
                0,
                min(self.position["y"], self.screen_height - self.size)
            )

        if self.wrap_cooldown > 0:
            self.wrap_cooldown -= 1

    # def shoot_missile(self) -> None:
    #     if len(self.missiles) == 0:
    #         missile_start_x = self.position["x"] + self.size // 2
    #         missile_start_y = self.position["y"] + self.size // 2
    #         missile = Missile(x=missile_start_x, y=missile_start_y, vx=5.0, vy=0.0)
    #         self.missiles.append(missile)
    #         logging.info("Training mode: Missile shot straight to the right.")
    #     else:
    #         logging.debug(
    #             "Attempted to shoot a missile in training mode, but one is already active."
    #         )
    # entities/player_training.py

    def shoot_missile(self, enemy_x: float, enemy_y: float) -> None:
        if len(self.missiles) == 0:  # Keep limit of 1 missile at a time for now
            missile_start_x = self.position["x"] + self.size // 2
            missile_start_y = self.position["y"] + self.size // 2

            # Calculate initial direction towards enemy
            dx = enemy_x - missile_start_x
            dy = enemy_y - missile_start_y
            angle = math.atan2(dy, dx)

            # Calculate velocity components
            speed = 5.0  # Or whatever base speed you want for missiles
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            # Add random variation to missile lifespan (optional, but good for training)
            lifespan = random.randint(500, 1500)  # Random lifespan 0.5s - 1.5s
            birth_time = pygame.time.get_ticks()  # For tracking lifespan

            missile = Missile(
                missile_start_x,
                missile_start_y,
                speed=speed,
                vx=vx,
                vy=vy,
                lifespan=lifespan,
                birth_time=birth_time,
            )
            self.missiles.append(missile)
            logging.info(
                f"Training Mode: Missile shot towards enemy at angle: {math.degrees(angle)}"
            )

    def update_missiles(self) -> None:
        for missile in self.missiles[:]:
            missile.update()
            # Remove if off-screen
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
