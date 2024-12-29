import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class MissileDataset(Dataset):
    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            # We now check for new keys: "dist" and "missile_collision"
            if (
                "player_x" in entry
                and "player_y" in entry
                and "enemy_x" in entry
                and "enemy_y" in entry
                and "missile_x" in entry
                and "missile_y" in entry
                and "missile_angle" in entry
                and "missile_action" in entry
                and "dist" in entry
                and "missile_collision" in entry
            ):
                # Convert the boolean collision to float
                collision_val = 1.0 if entry["missile_collision"] else 0.0

                # Build a 9-element state
                state = [
                    entry["player_x"],
                    entry["player_y"],
                    entry["enemy_x"],
                    entry["enemy_y"],
                    entry["missile_x"],
                    entry["missile_y"],
                    entry["missile_angle"],
                    entry["dist"],  # new
                    collision_val,  # new
                ]
                action = [entry["missile_action"]]
                self.samples.append((state, action))
            # If you want to handle missing fields by fallback, do something like:
            # else:
            #     # Provide defaults if 'dist' or 'missile_collision' is missing
            #     dist_val = entry.get("dist", 0.0)
            #     collision_val = 1.0 if entry.get("missile_collision", False) else 0.0
            #     state = [...]
            #     action = [...]
            #     self.samples.append((state, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, action = self.samples[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
        )


class SimpleMissileModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=1):
        super(SimpleMissileModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(filename):
    dataset = MissileDataset(filename)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Updated model: input_size=9
    model = SimpleMissileModel(input_size=9, hidden_size=64, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):  # 20 epochs, adjust as needed
        for states, actions in dataloader:
            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, actions)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model weights
    torch.save(model.state_dict(), "models/missile_model.pth")


if __name__ == "__main__":
    train_model("data/raw/training_data.json")
