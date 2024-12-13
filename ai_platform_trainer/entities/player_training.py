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
        Update the player's position each frame during training mode.
        The player will try to maintain a good distance from the enemy by moving away from it.
        """

        # Desired minimum distance from enemy
        desired_distance = 200  # Adjust this value as needed

        px, py = self.position["x"], self.position["y"]
        dx = px - enemy_x
        dy = py - enemy_y
        dist = (dx**2 + dy**2)**0.5

        if dist < desired_distance:
            # Enemy is too close, move away
            if dist > 0:  # Avoid division by zero
                dx /= dist
                dy /= dist
            else:
                # If exactly on the enemy, move randomly to escape
                dx, dy = random.choice([(-1,0),(1,0),(0,-1),(0,1)])

            # Move away from the enemy
            self.position["x"] += dx * self.step
            self.position["y"] += dy * self.step
        else:
            # Enemy is sufficiently far; move randomly or stay still
            # Add a small random movement so the player isn't static:
            rand_dx = random.choice([-1, 0, 1]) * self.step
            rand_dy = random.choice([-1, 0, 1]) * self.step
            self.position["x"] += rand_dx
            self.position["y"] += rand_dy

        # Wrap-around logic to keep player on screen
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
