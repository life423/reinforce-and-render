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
            if (
                "missile_x" in entry
                and "missile_y" in entry
                and "missile_action" in entry
            ):
                # Extract state and action
                state = [
                    entry["player_x"],
                    entry["player_y"],
                    entry["enemy_x"],
                    entry["enemy_y"],
                    entry["missile_x"],
                    entry["missile_y"],
                    entry["missile_angle"],
                ]
                action = [entry["missile_action"]]
                self.samples.append((state, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, action = self.samples[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(
            action, dtype=torch.float32
        )


# Simple model
class SimpleMissileModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, output_size=1):
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

    model = SimpleMissileModel()
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

    torch.save(model.state_dict(), "missile_model.pth")


if __name__ == "__main__":
    train_model("data/raw/training_data.json")
