# ai_platform_trainer/entities/player_training.py
import random
import pygame
import logging
import random
from ai_platform_trainer.entities.missile import Missile


class PlayerTraining:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue or any color you prefer
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.step = 5
        self.missiles = []
        logging.info("PlayerTraining initialized.")

    def reset(self) -> None:
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.missiles.clear()
        logging.info("PlayerTraining has been reset.")

    def update(self, enemy_x: float, enemy_y: float) -> None:
        """
        Update the player state each frame during training mode.
        Let's move the player randomly to ensure it's not stationary.
        """

        # Random horizontal and vertical steps: -1, 0, or 1 times self.step
        dx = random.choice([-1, 0, 1]) * self.step
        dy = random.choice([-1, 0, 1]) * self.step

        self.position["x"] += dx
        self.position["y"] += dy

        # Wrap-around logic or boundary checks if you want the player to stay on-screen
        if self.position["x"] < -self.size:
            self.position["x"] = self.screen_width
        elif self.position["x"] > self.screen_width:
            self.position["x"] = -self.size

        if self.position["y"] < -self.size:
            self.position["y"] = self.screen_height
        elif self.position["y"] > self.screen_height:
            self.position["y"] = -self.size

    def shoot_missile(self) -> None:
        missile_start_x = self.position["x"] + self.size // 2
        missile_start_y = self.position["y"] + self.size // 2
        missile = Missile(x=missile_start_x, y=missile_start_y, vx=5.0, vy=0.0)
        self.missiles.append(missile)
        logging.info("Training mode: Missile shot straight to the right.")

    def update_missiles(self, enemy_pos: tuple[int, int]) -> None:
        ex, ey = enemy_pos
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
        # Draw the player as a rectangle
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
        self.draw_missiles(screen)
