import torch
import pygame
import random
import os
from ai_model.model import EnemyAIModel


class Enemy:
    def __init__(
        self,
        start_x: int,
        start_y: int,
        screen_width: int,
        screen_height: int,
        mode="play",
        model_path="ai_model/saved_models/enemy_ai_model.pth",
    ) -> None:
        """
        Initialize the enemy with a starting position and screen constraints.
        """
        self.start_x = start_x  # Store starting x position
        self.start_y = start_y  # Store starting y position
        self.pos = {"x": start_x, "y": start_y}
        self.size = 100  # Size of the enemy block
        self.color = (255, 69, 0)  # Orange-Red color for the enemy
        self.speed = max(1, screen_width // 500)  # Speed of enemy movement
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.movement_counter = 0
        self.mode = (
            mode  # Mode can be 'training' or 'play' to dictate movement behavior
        )
        self.current_direction = random.choice(
            [(-self.speed, 0), (self.speed, 0), (0, -self.speed), (0, self.speed)]
        )

        # Load the AI model if a path is provided and mode is "play"
        self.ai_model = None
        # Use the default model path if none is provided
        if not model_path:
            model_path = "ai_model/saved_models/enemy_ai_model.pth"
        if model_path and mode == "play":
            input_size = 4  # Assuming the model takes player_x, player_y, enemy_x, enemy_y as inputs
            hidden_size = 64
            output_size = 1
            self.ai_model = EnemyAIModel(input_size, hidden_size, output_size)
            self.ai_model.load_state_dict(torch.load(model_path))
            self.ai_model.eval()  # Set the model to evaluation mode

    def reset_position(self) -> None:
        """
        Reset the enemy's position to its starting position.
        """
        self.pos = {"x": self.start_x, "y": self.start_y}
        self.movement_counter = 0  # Reset movement counter if necessary

    def update_position(self, player_pos: dict) -> None:
        """
        Update the position of the enemy based on the player's position.
        The movement is varied to create more dynamic behavior.

        Args:
            player_pos (dict): The current position of the player.
        """
        if self.mode == "play" and self.ai_model:
            # Use the AI model to determine movement towards the player
            input_tensor = torch.tensor(
                [[player_pos["x"], player_pos["y"], self.pos["x"], self.pos["y"]]],
                dtype=torch.float32,
            )

            with torch.no_grad():
                distance = self.ai_model(input_tensor).item()

            # Example logic to move towards the player based on model output
            if self.pos["x"] < player_pos["x"]:
                self.pos["x"] += min(self.speed, distance)
            elif self.pos["x"] > player_pos["x"]:
                self.pos["x"] -= min(self.speed, distance)

            if self.pos["y"] < player_pos["y"]:
                self.pos["y"] += min(self.speed, distance)
            elif self.pos["y"] > player_pos["y"]:
                self.pos["y"] -= min(self.speed, distance)
        else:
            # Change direction every 20 frames to make movement more varied
            if self.movement_counter % 20 == 0:
                self.current_direction = random.choice(
                    [
                        (-self.speed, 0),
                        (self.speed, 0),
                        (0, -self.speed),
                        (0, self.speed),
                    ]
                )
            dx, dy = self.current_direction

            new_x = self.pos["x"] + dx
            new_y = self.pos["y"] + dy

            # Ensure the enemy does not move off-screen
            self.pos["x"] = max(0, min(self.screen_width - self.size, new_x))
            self.pos["y"] = max(0, min(self.screen_height - self.size, new_y))

        # Increment movement counter
        self.movement_counter += 1

    def draw(self, screen) -> None:
        """
        Draw the enemy on the screen.
        """
        pygame.draw.rect(
            screen, self.color, (self.pos["x"], self.pos["y"], self.size, self.size)
        )
