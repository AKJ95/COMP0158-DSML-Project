import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim


class EncoderDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    x_encoder_vectors = np.load('data/processed/x_encoder_vectors.npy')
    x_encoder_labels = np.load('data/processed/x_encoder_labels.npy')

    # Convert numpy arrays to PyTorch tensors
    x_encoder_vectors = torch.from_numpy(x_encoder_vectors)
    x_encoder_labels = torch.from_numpy(x_encoder_labels)

    dataset = EncoderDataset(x_encoder_vectors, x_encoder_labels)

    # Create DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Instantiate the MLP
    model = MLP()

    # Check if CUDA is available and if so, set the device to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)

    model.train()

    # Define the loss function and the optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the number of epochs
    n_epochs = 10

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for i, (vectors, labels) in enumerate(data_loader):
            # Move the data to the device
            vectors = vectors.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(vectors)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * vectors.size(0)

            # Print loss every 100 batches
            if i % 100 == 0:
                print(f'Epoch {epoch + 1}/{n_epochs}, Step {i}/{len(data_loader)}, Loss: {loss.item()}')

        # Calculate average losses
        train_loss = train_loss / len(data_loader.dataset)

        # Print training statistics
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
