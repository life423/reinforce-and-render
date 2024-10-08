import torch
import random

class Enemy:
    def __init__(self, start_x: int, start_y: int, screen_width: int, screen_height: int, model_path: str = None) -> None:
        """
        Initialize the enemy with a starting position and screen constraints.
        """
        self.pos = {'x': start_x, 'y': start_y}
        self.size = 100  # Size of the enemy block
        self.color = (255, 69, 0)  # Orange-Red color for the enemy
        self.speed = 5  # Default speed, can be adjusted
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.model = None

        if model_path:
            # Load the trained AI model if path is provided
            input_size = 4  # Player (x, y) and Enemy (x, y)
            hidden_size = 64
            output_size = 1

            self.model = EnemyAIModel(input_size, hidden_size, output_size)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def update_position(self, player_pos: dict) -> None:
        """
        Update the position of the enemy based on the player's position using the trained AI model.

        Args:
            player_pos (dict): The current position of the player.
        """
        if self.model:
            # Convert positions to tensor for the model
            input_data = torch.tensor([player_pos['x'], player_pos['y'], self.pos['x'], self.pos['y']], dtype=torch.float32)
            output = self.model(input_data)  # Forward pass through the model

            # The output will determine the movement direction
            movement_choice = int(output.item()) % 4  # Map the output to one of four directions

            directions = [
                (-self.speed, 0),  # left
                (self.speed, 0),   # right
                (0, -self.speed),  # up
                (0, self.speed)    # down
            ]

            dx, dy = directions[movement_choice]
        else:
            # Fallback to random movement if no model is provided
            directions = [
                (-self.speed, 0),  # left
                (self.speed, 0),   # right
                (0, -self.speed),  # up
                (0, self.speed)    # down
            ]
            dx, dy = random.choice(directions)

        # Update the enemy position
        new_x = self.pos['x'] + dx
        new_y = self.pos['y'] + dy

        # Ensure the enemy does not move off-screen
        self.pos['x'] = max(0, min(self.screen_width - self.size, new_x))
        self.pos['y'] = max(0, min(self.screen_height - self.size, new_y))

    def draw(self, screen) -> None:
        """
        Draw the enemy on the screen.
        """
        pygame.draw.rect(screen, self.color, (self.pos['x'], self.pos['y'], self.size, self.size))
