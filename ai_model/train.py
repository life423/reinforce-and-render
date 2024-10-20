from ai_model.model_definition.model import EnemyAIModel
import torch
import torch.optim as optim


def train():
    # Define the training parameters
    input_size = 4  # Example: [enemy_x, enemy_y, player_x, player_y]
    hidden_size = 16
    output_size = 2  # Example: [dx, dy] directions
    model = EnemyAIModel(input_size, hidden_size, output_size)

    # Define an optimizer and a loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()  # Using Mean Squared Error for simplicity

    # Training loop (placeholder for actual training logic)
    for epoch in range(100):  # Example: 100 epochs
        # Placeholder: Generate random training data for input and output
        input_tensor = torch.randn((1, input_size))
        target_tensor = torch.randn((1, output_size))

        # Forward pass
        output = model(input_tensor)

        # Compute loss
        loss = criterion(output, target_tensor)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/100], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'ai_model/saved_models/enemy_ai_model.pth')


if __name__ == "__main__":
    train()
