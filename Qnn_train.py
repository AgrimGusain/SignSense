import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Load_data import NoiseDataset  # Import NoiseDataset
from quantum_preprocessor import DenoiseQNN  # Import DenoiseQNN

def train():
    # Set the directories to the noisy and clean datasets
    dataset = NoiseDataset("dataset/noisy", "dataset/clean")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instantiate the model, optimizer, and loss function
    model = DenoiseQNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(5):
        total_loss = 0
        for noisy, clean, labels in dataloader:
            # Ensure the input shape is correct
            if noisy.shape[1:] != (3, 32, 32):
                noisy = noisy.view(noisy.size(0), 3, 32, 32)  # Reshape if necessary

            print(f"Input shape to model: {noisy.shape}")
            print(f"Training on batch with labels: {set(labels)}")  # Unique alphabets in the batch

            output = model(noisy)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "qnn_denoiser.pth")

if __name__ == "__main__":
    train()
