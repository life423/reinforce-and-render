import pygame
from typing import List, Tuple
from ai_platform_trainer.entities.missile import Missile
import logging


class PlayerPlay:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 255, 0)  # Green
        self.position = {"x": screen_width // 4, "y": screen_height // 2}
        self.step = 5
        self.missiles: List[Missile] = []

        # Initialize other player attributes as needed

    def reset(self) -> None:
        """
        Reset the player's position and other attributes to their initial state.
        """
        self.position = {"x": self.screen_width // 4, "y": self.screen_height // 2}
        self.missiles.clear()  # Clear any active missiles
        logging.info("Player has been reset to the initial position.")

    def handle_input(self) -> bool:
        """
        Handle player input for movement.

        :return: False if player quits, True otherwise.
        """
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.position["x"] -= self.step
        if keys[pygame.K_RIGHT]:
            self.position["x"] += self.step
        if keys[pygame.K_UP]:
            self.position["y"] -= self.step
        if keys[pygame.K_DOWN]:
            self.position["y"] += self.step

        # Keep player within screen bounds
        self.position["x"] = max(
            0, min(self.screen_width - self.size, self.position["x"])
        )
        self.position["y"] = max(
            0, min(self.screen_height - self.size, self.position["y"])
        )

        return True  # Continue running

    def shoot_missile(self, target_x: int, target_y: int) -> None:
        """Shoot a missile towards the target coordinates."""
        missile = Missile(
            x=self.position["x"] + self.size // 2,
            y=self.position["y"] + self.size // 2,
            target_x=target_x,
            target_y=target_y,
        )
        self.missiles.append(missile)
        logging.info(f"Missile shot towards ({target_x}, {target_y}).")

    def update_missiles(self, enemy_pos: Tuple[int, int]) -> None:
        """Update all active missiles."""
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
        """Draw all active missiles."""
        for missile in self.missiles:
            missile.draw(screen)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the player on the screen."""
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
        self.draw_missiles(screen)
