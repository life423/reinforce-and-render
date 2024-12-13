import torch
import math
import pygame
import logging
from typing import Optional, Tuple


class Enemy:
    def __init__(self, screen_width: int, screen_height: int, model):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        # Updated enemy color to a bright gold
        self.color = (255, 215, 0)  # Gold
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.model = model
        self.base_speed = max(2, screen_width // 400)
        self.visible = True

        # Fade-in attributes
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_duration = 300  # milliseconds
        self.fade_start_time = 0

        # Create a Surface for the enemy with per-pixel alpha
        self.surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.surface.fill((*self.color, 255))

    def update_movement(
        self, player_x: float, player_y: float, player_speed: int, current_time: int
    ):
        if not self.visible:
            return

        # State: (player_x, player_y, enemy_x, enemy_y, dist)
        dist = math.sqrt(
            (player_x - self.pos["x"]) ** 2 + (player_y - self.pos["y"]) ** 2
        )
        state = torch.tensor(
            [[player_x, player_y, self.pos["x"], self.pos["y"], dist]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            action = self.model(state)

        action_dx, action_dy = action[0].tolist()

        # Normalize action vector
        action_len = math.sqrt(action_dx**2 + action_dy**2)
        if action_len > 0:
            action_dx /= action_len
            action_dy /= action_len
        else:
            action_dx, action_dy = 0.0, 0.0

        speed = player_speed * 0.7
        self.pos["x"] += action_dx * speed
        self.pos["y"] += action_dy * speed

        # Instead of wrap-around, clamp the enemy to stay within screen bounds
        self.pos["x"] = max(0, min(self.screen_width - self.size, self.pos["x"]))
        self.pos["y"] = max(0, min(self.screen_height - self.size, self.pos["y"]))

    def draw(self, screen: pygame.Surface) -> None:
        if self.visible:
            self.surface.set_alpha(self.fade_alpha)
            screen.blit(self.surface, (self.pos["x"], self.pos["y"]))

    def hide(self) -> None:
        self.visible = False
        logging.info("Enemy hidden due to collision.")

    def show(self, current_time: int) -> None:
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time
        logging.info("Enemy set to fade in.")

    def update_fade_in(self, current_time: int) -> None:
        if self.fading_in:
            elapsed = current_time - self.fade_start_time
            if elapsed >= self.fade_duration:
                self.fade_alpha = 255
                self.fading_in = False
                logging.info("Enemy fade-in completed.")
            else:
                self.fade_alpha = int((elapsed / self.fade_duration) * 255)
                logging.debug(f"Enemy fade-in alpha: {self.fade_alpha}")

    def set_position(self, x: int, y: int) -> None:
        self.pos["x"], self.pos["y"] = x, y
