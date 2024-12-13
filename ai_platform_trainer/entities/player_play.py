import pygame
from typing import List, Tuple
from ai_platform_trainer.entities.missile import Missile
import logging


class PlayerPlay:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue
        self.position = {"x": screen_width // 4, "y": screen_height // 2}
        self.step = 5
        self.missiles: List[Missile] = []

    def reset(self) -> None:
        self.position = {"x": self.screen_width // 4, "y": self.screen_height // 2}
        self.missiles.clear()
        logging.info("Player has been reset to the initial position.")

    def handle_input(self) -> bool:
        keys = pygame.key.get_pressed()

        # WASD and Arrow Keys movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] -= self.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] += self.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] -= self.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] += self.step

        # Wrap-around logic for the player
        if self.position["x"] < -self.size:
            self.position["x"] = self.screen_width
        elif self.position["x"] > self.screen_width:
            self.position["x"] = -self.size

        if self.position["y"] < -self.size:
            self.position["y"] = self.screen_height
        elif self.position["y"] > self.screen_height:
            self.position["y"] = -self.size

        return True

    def shoot_missile(self) -> None:
        """Shoot a missile directly to the right of the player's position."""
        missile_start_x = self.position["x"] + self.size // 2
        missile_start_y = self.position["y"] + self.size // 2
        # Create a missile that travels in a straight line (to the right)
        missile = Missile(x=missile_start_x, y=missile_start_y, vx=5.0, vy=0.0)
        self.missiles.append(missile)
        logging.info("Missile shot straight to the right.")

    def update_missiles(self, enemy_pos: Tuple[int, int]) -> None:
        for missile in self.missiles[:]:
            missile.update()
            # Remove missile if it goes off-screen
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
