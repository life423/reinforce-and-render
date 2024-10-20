import torch  # Importing PyTorch library, which is used for building and training neural networks
# Importing the 'nn' module from PyTorch for creating neural network layers
import torch.nn as nn
# Importing functional API for activation functions and other utilities
import torch.nn.functional as F


class EnemyAIModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the EnemyAIModel neural network.

        Args:
            input_size (int): Number of input features (e.g., positions of player and enemy).
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output features (e.g., possible actions for enemy).
        """
        super(EnemyAIModel, self).__init__(
        )  # Call the initializer of the parent class (nn.Module)
        # Define the first fully connected layer from input to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Define the second fully connected layer from hidden to output layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor representing the current state of the game.

        Returns:
            torch.Tensor: Output tensor representing the predicted action.
        """
        x = F.relu(
            self.fc1(x))  # Apply ReLU activation function to the output of the first layer
        # Pass the output from the hidden layer to the output layer
        x = self.fc2(x)
        return x  # Return the final output
