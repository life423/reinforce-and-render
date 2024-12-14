# ai_platform_trainer/entities/player_training.py
import random
import pygame
import logging
import math
from ai_platform_trainer.entities.missile import Missile


class PlayerTraining:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.step = 5
        self.missiles = []
        logging.info("PlayerTraining initialized.")

        self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.direction_timer = 0
        self.current_direction = (0, 0)
        self.currently_moving_away = False

        # Define hysteresis parameters
        self.desired_distance = 200
        self.margin = 20

        self.pick_new_direction()

    def pick_new_direction(self) -> None:
        self.current_direction = random.choice(self.directions)
        self.direction_timer = random.randint(60, 180)

    def reset(self) -> None:
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.missiles.clear()
        self.pick_new_direction()
        logging.info("PlayerTraining has been reset.")

    def move_away_from(self, enemy_x: float, enemy_y: float) -> None:
        px, py = self.position["x"], self.position["y"]
        dx, dy = px - enemy_x, py - enemy_y
        dist = math.hypot(dx, dy)
        if dist > 0:
            ndx, ndy = dx / dist, dy / dist
        else:
            ndx, ndy = random.choice(self.directions)
        self.position["x"] += ndx * self.step
        self.position["y"] += ndy * self.step
        self.currently_moving_away = True

    def move_random_direction(self) -> None:
        if self.direction_timer <= 0:
            self.pick_new_direction()
        ndx, ndy = self.current_direction
        self.position["x"] += ndx * self.step
        self.position["y"] += ndy * self.step
        self.direction_timer -= 1
        self.currently_moving_away = False

    def is_currently_moving_away(self) -> bool:
        return self.currently_moving_away

    def update(self, enemy_x: float, enemy_y: float) -> None:
        dist = math.hypot(self.position["x"] - enemy_x, self.position["y"] - enemy_y)

        if dist < self.desired_distance - self.margin:
            self.move_away_from(enemy_x, enemy_y)
        elif dist > self.desired_distance + self.margin:
            self.move_random_direction()
        else:
            if self.is_currently_moving_away():
                self.move_away_from(enemy_x, enemy_y)
            else:
                self.move_random_direction()

        self.position["x"] %= self.screen_width
        self.position["y"] %= self.screen_height

    def shoot_missile(self) -> None:
        missile_start_x = self.position["x"] + self.size // 2
        missile_start_y = self.position["y"] + self.size // 2
        missile = Missile(x=missile_start_x, y=missile_start_y, vx=5.0, vy=0.0)
        self.missiles.append(missile)
        logging.info("Training mode: Missile shot straight to the right.")

    def update_missiles(self, enemy_pos: tuple[int, int]) -> None:
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
