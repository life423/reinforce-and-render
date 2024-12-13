import torch
import math
import pygame


class Enemy:
    def __init__(self, screen_width, screen_height, model):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (173, 153, 228)
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.model = model
        self.base_speed = max(2, screen_width // 400)
        self.visible = True  # New attribute to control visibility

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
            pygame.draw.rect(
                screen, self.color, (self.pos["x"], self.pos["y"], self.size, self.size)
            )

    def hide(self):
        """Hide the enemy by setting visibility to False."""
        self.visible = False

    def show(self):
        """Show the enemy by setting visibility to True."""
        self.visible = True

    def set_position(self, x: int, y: int):
        """Set the enemy's position."""
        self.pos["x"], self.pos["y"] = x, y
