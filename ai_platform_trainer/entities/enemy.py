import torch
import math

class Enemy:
    def __init__(self, screen_width, screen_height, model):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        # Your chosen color
        self.color = (173, 153, 228)
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.model = model  # Store the model
        self.base_speed = max(2, screen_width // 400)

    def update_movement(self, player_x, player_y, player_speed):
        # Construct the state vector
        state = torch.tensor([[player_x, player_y, self.pos["x"], self.pos["y"]]], dtype=torch.float32)

        # Get action from model
        with torch.no_grad():
            action = self.model(state)  # shape: [1,2]
        
        action_dx, action_dy = action[0].tolist()

        # Move enemy according to model's predicted action
        # You can also scale action_dx, action_dy by some factor if needed
        speed = player_speed * 0.7
        # Normalize if needed
        dist = math.sqrt(action_dx**2 + action_dy**2)
        if dist > 0:
            action_dx /= dist
            action_dy /= dist
        self.pos["x"] += action_dx * speed
        self.pos["y"] += action_dy * speed

        # Clamp or wrap-around
        self.pos["x"] = max(0, min(self.screen_width - self.size, self.pos["x"]))
        self.pos["y"] = max(0, min(self.screen_height - self.size, self.pos["y"]))

    def draw(self, screen):
        # Draw as before
        import pygame
        pygame.draw.rect(screen, self.color, (self.pos["x"], self.pos["y"], self.size, self.size))