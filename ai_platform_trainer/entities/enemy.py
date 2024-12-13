import torch
import math
import pygame
import logging


class Enemy:
    def __init__(self, screen_width, screen_height, model):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (173, 153, 228)
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.model = model
        self.base_speed = max(2, screen_width // 400)
        self.visible = True  # Controls rendering
        self.fading_in = False  # Indicates if fade-in is active
        self.fade_start_time = 0  # Time when fade-in starts
        self.alpha = 255  # Transparency level (0-255)
        self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.image.fill(self.color + (self.alpha,))  # Initial full opacity

    def wrap_position(self, x, y):
        if x < -self.size:
            x = self.screen_width
        elif x > self.screen_width:
            x = -self.size
        if y < -self.size:
            y = self.screen_height
        elif y > self.screen_height:
            y = -self.size
        return x, y

    def update_movement(self, player_x, player_y, player_speed):
        if not self.visible:
            return  # Don't update movement if enemy is not visible

        # State with 5 inputs: (player_x, player_y, self.pos["x"], self.pos["y"], dist)
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
            action_dx, action_dy = 0.0, 0.0  # Prevent division by zero

        speed = player_speed * 0.7
        self.pos["x"] += action_dx * speed
        self.pos["y"] += action_dy * speed

        # Wrap-around logic
        self.pos["x"], self.pos["y"] = self.wrap_position(self.pos["x"], self.pos["y"])

    def draw(self, screen):
        if self.visible:
            self.image.set_alpha(self.alpha)
            screen.blit(self.image, (self.pos["x"], self.pos["y"]))

    def hide(self):
        """Hide the enemy by setting visibility to False."""
        self.visible = False
        logging.info("Enemy hidden due to collision.")

    def show(self):
        """Show the enemy by setting visibility to True and starting fade-in."""
        self.visible = True
        self.start_fade_in(pygame.time.get_ticks())

    def set_position(self, x: int, y: int):
        """Set the enemy's position."""
        self.pos["x"], self.pos["y"] = x, y

    def start_fade_in(self, current_time: int):
        """Initiate the fade-in effect."""
        self.fading_in = True
        self.fade_start_time = current_time
        self.alpha = 0  # Start fully transparent
        self.image.set_alpha(self.alpha)
        logging.info("Enemy fade-in started.")

    def update_fade_in(self, current_time: int, fade_duration: int = 300):
        """
        Update the fade-in effect based on the elapsed time.

        :param current_time: Current time in milliseconds.
        :param fade_duration: Duration of the fade-in in milliseconds.
        """
        if self.fading_in:
            elapsed = current_time - self.fade_start_time
            if elapsed >= fade_duration:
                self.alpha = 255  # Fully opaque
                self.fading_in = False
                logging.info("Enemy fade-in completed.")
            else:
                # Calculate current alpha based on elapsed time
                self.alpha = int((elapsed / fade_duration) * 255)
                logging.debug(f"Enemy fade-in alpha updated to {self.alpha}.")
            self.image.set_alpha(self.alpha)
