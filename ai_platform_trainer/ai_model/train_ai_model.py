import torch
import torch.nn as nn
import torch.optim as optim

import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
mongo_uri = os.getenv('MONGO_URI')

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client['pixel_pursuit_db']
collection = db['training_data']

# Define the model


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)  # Output a single value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Load data from MongoDB


def load_data():
    data = list(collection.find())
    X = []
    y = []

    for item in data:
        X.append([
            item['player_position']['x'],
            item['player_position']['y'],
            item['enemy_position']['x'],
            item['enemy_position']['y']
        ])
        y.append([item['collision']])

    return np.array(X), np.array(y)


# Prepare the data
X, y = load_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Initialize model, loss, and optimiz

# Initialize model, loss, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()  # Mean Squared Error if you're predicting a scalar
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):  # Adjust epochs as needed
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/1000], Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'saved_models/enemy_ai_model.pth')
