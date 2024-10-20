from ai_model.model_definition.model import EnemyAIModel
import torch


class Enemy:
    def __init__(self, screen_width, screen_height):
        self.pos = {"x": screen_width // 2, "y": screen_height // 2}
        self.size = 100
        self.color = (255, 69, 0)
        self.speed = max(1, screen_width // 500)
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Initialize the AI model
        # Example input size and hidden/output sizes
        self.ai_model = EnemyAIModel(
            input_size=4, hidden_size=16, output_size=2)
        self.ai_model.eval()  # Set the model to evaluation mode

    def update(self, player_pos):
        # Using AI to update position
        # Prepare input tensor with enemy and player positions
        input_tensor = torch.tensor(
            [self.pos["x"], self.pos["y"], player_pos["x"], player_pos["y"]], dtype=torch.float32)

        # Get the output from the AI model
        output = self.ai_model(input_tensor)

        # Use the output to decide movement direction
        # Assuming output is something like [dx, dy] representing direction changes
        self.pos["x"] += int(output[0].item()) * self.speed
        self.pos["y"] += int(output[1].item()) * self.speed

        # Clamp position to stay on the screen
        self.pos["x"] = max(
            0, min(self.screen_width - self.size, self.pos["x"]))
        self.pos["y"] = max(
            0, min(self.screen_height - self.size, self.pos["y"]))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color,
                         (self.pos["x"], self.pos["y"], self.size, self.size))
