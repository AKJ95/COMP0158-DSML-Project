import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset


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
