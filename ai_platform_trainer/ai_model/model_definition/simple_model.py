import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=2, dropout_prob=0.2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # Expect x to be (batch_size, 5) since we added distance
        if x.dim() != 2 or x.size(1) != self.fc1.in_features:
            raise ValueError(
                f"Expected input to have shape (batch_size, {self.fc1.in_features})."
            )

        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
