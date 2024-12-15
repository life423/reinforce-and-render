import json
import torch
from torch.utils.data import Dataset


class MissileDataset(Dataset):
    def __init__(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        self.samples = []
        for entry in data:
            if (
                "missile_x" in entry
                and "missile_y" in entry
                and "missile_action" in entry
            ):
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
