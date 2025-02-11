# file: ai_platform_trainer/ai_model/missile_dataset.py
import json
import torch
from torch.utils.data import Dataset

class MissileDataset(Dataset):
    """
    PyTorch dataset for missile training. Reads data from a JSON file
    and returns (state, action, weight) tuples.
    """
    def __init__(self, filename: str):
        super().__init__()
        with open(filename, "r") as f:
            self.data = json.load(f)

        self.samples = []
        self.weights = []

        needed_keys = [
            "player_x", "player_y", "enemy_x", "enemy_y",
            "missile_x", "missile_y", "missile_angle",
            "dist", "missile_collision", "missile_action",
        ]
        for entry in self.data:
            if all(k in entry for k in needed_keys):
                collision_val = 1.0 if entry["missile_collision"] else 0.0
                state = [
                    entry["player_x"],
                    entry["player_y"],
                    entry["enemy_x"],
                    entry["enemy_y"],
                    entry["missile_x"],
                    entry["missile_y"],
                    entry["missile_angle"],
                    entry["dist"],
                    collision_val,
                ]
                action = [entry["missile_action"]]
                wt = 2.0 if collision_val == 1.0 else 1.0
                self.samples.append((state, action))
                self.weights.append(wt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        state, action = self.samples[idx]
        weight = self.weights[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )