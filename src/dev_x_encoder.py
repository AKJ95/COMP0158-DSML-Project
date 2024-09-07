import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall


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
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    x_encoder_vectors = np.load('data/processed/x_encoder_vectors_ens_train.npy')
    x_encoder_labels = np.load('data/processed/x_encoder_labels_ens_train.npy')
    x_encoder_vectors_dev = np.load('data/processed/x_encoder_vectors_ens_test.npy')
    x_encoder_labels_dev = np.load('data/processed/x_encoder_labels_ens_test.npy')

    # Convert numpy arrays to PyTorch tensors
    x_encoder_vectors = torch.from_numpy(x_encoder_vectors).float()
    x_encoder_labels = torch.from_numpy(x_encoder_labels).float()
    x_encoder_labels = torch.unsqueeze(x_encoder_labels, 1)
    x_encoder_vectors_dev = torch.from_numpy(x_encoder_vectors_dev).float()
    x_encoder_labels_dev = torch.from_numpy(x_encoder_labels_dev).float()
    x_encoder_labels_dev = torch.unsqueeze(x_encoder_labels_dev, 1)

    dataset = EncoderDataset(x_encoder_vectors, x_encoder_labels)
    dataset_dev = EncoderDataset(x_encoder_vectors_dev, x_encoder_labels_dev)

    # Create DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dataset_dev, batch_size=64, shuffle=False)

    # Instantiate the MLP
    model = MLP()

    # Check if CUDA is available and if so, set the device to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)

    model.load_state_dict(torch.load('models/xencoder/x_encoder_model_ens.pt'))

    best_dev_loss = np.inf

    # Training loop
    dev_labels = np.array([])
    dev_preds = np.array([])
    model.eval()
    dev_loss = 0.0
    for i, (vectors, labels) in enumerate(dev_loader):
        vectors = vectors.to(device)
        labels = labels.to(device)

        outputs = model(vectors)
        # if i == 0:
        #     dev_preds = torch.sigmoid(outputs).cpu().detach().numpy()
        #     dev_labels = labels.cpu().detach().numpy().astype(int)
        dev_preds = np.append(dev_preds, torch.sigmoid(outputs).cpu().detach().numpy())
        dev_labels = np.append(dev_labels, labels.cpu().detach().numpy())
    print(dev_preds[:10])
    print(len(dev_preds))
    print(len(dev_labels))

    mention_count = 0
    correct_count = 0
    current_correct_prob = 0.0
    max_probability = 0.0
    for i in range(len(dev_preds)):
        if dev_preds[i] > max_probability:
            max_probability = dev_preds[i]
        if dev_labels[i] == 1:
            current_correct_prob = dev_preds[i]
        if i % 4 == 3:
            mention_count += 1
            if current_correct_prob == max_probability:
                correct_count += 1
            current_correct_prob = 0.0
            max_probability = 0.0

    dev_loss = dev_loss / len(dev_loader.dataset)
    dev_preds = torch.from_numpy(dev_preds)
    dev_labels = torch.from_numpy(dev_labels).int()
    threshold = 0.75
    accuracy = BinaryAccuracy(threshold=threshold)
    precision = BinaryPrecision(threshold=threshold)
    recall = BinaryRecall(threshold=threshold)
    accuracy.update(dev_preds, dev_labels)
    precision.update(dev_preds, dev_labels)
    recall.update(dev_preds, dev_labels)
    print(f'Validation Accuracy: {accuracy.compute()}')
    print(f'Validation Precision: {precision.compute()}')
    print(f'Validation Recall: {recall.compute()}')
    print(f'Validation Loss: {dev_loss}')
    print(f'Label distribution: {1 - torch.sum(dev_labels)/len(dev_labels)}% are negative')
