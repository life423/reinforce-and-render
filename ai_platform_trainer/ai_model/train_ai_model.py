import torch
import torch.nn as nn
import torch.optim as optim
import json
import math
from ai_platform_trainer.ai_model.model_definition.model import SimpleModel


def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    states = []
    actions = []
    for step in data:
        # Extract positions
        px = step["player_x"]
        py = step["player_y"]
        ex = step["enemy_x"]
        ey = step["enemy_y"]

        # Compute distance if desired
        dist = math.sqrt((px - ex) ** 2 + (py - ey) ** 2)

        # For now, let's just use (player_x, player_y, enemy_x, enemy_y) as the state.
        # If you want to include distance, you'd add it to the state vector.
        # Example if including distance: state = [px, py, ex, ey, dist]
        # Then input_size in the model would be 5.

        state = [px, py, ex, ey]  # input_size=4 as defined
        # Extract the action
        adx = step["action_dx"]
        ady = step["action_dy"]

        # If doing supervised learning to predict action_dx, action_dy:
        # collision could be used to filter data if you want, but we won't here.
        actions.append([adx, ady])
        states.append(state)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    return states, actions


def train_model(states, actions, epochs=50, lr=1e-3):
    model = SimpleModel(input_size=4, hidden_size=64, output_size=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save the model weights
    torch.save(model.state_dict(), "models/enemy_ai_model.pth")


if __name__ == "__main__":
    states, actions = load_data("data/raw/training_data.json")
    train_model(states, actions, epochs=50, lr=1e-3)
