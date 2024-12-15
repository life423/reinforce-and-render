import torch
import torch.nn as nn


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
