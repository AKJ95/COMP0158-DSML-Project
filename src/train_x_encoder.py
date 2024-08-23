import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
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
    x_encoder_vectors = np.load('data/processed/x_encoder_vectors.npy')
    x_encoder_labels = np.load('data/processed/x_encoder_labels.npy')
    x_encoder_vectors_dev = np.load('data/processed/x_encoder_vectors_dev.npy')
    x_encoder_labels_dev = np.load('data/processed/x_encoder_labels_dev.npy')

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

    model.train()

    # Define the loss function and the optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Define the number of epochs
    n_epochs = 100
    train_losses = []

    best_dev_loss = np.inf

    # Training loop
    for epoch in range(n_epochs):
        dev_labels = np.array([])
        dev_preds = np.array([])
        instance_counter = 0
        model.train()
        train_loss = 0.0
        for i, (vectors, labels) in enumerate(data_loader):
            # Move the data to the device
            vectors = vectors.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(vectors)
            loss = criterion(outputs, labels)

            # Print model outputs for the first 5 instances
            # if instance_counter < 10:
            #     for j in range(vectors.size(0)):
            #         if instance_counter < 10:
            #             print(f'Model outputs for instance {instance_counter + 1}: {torch.sigmoid(outputs[j]).item()}; Actual label: {labels[j].item()}')
            #             instance_counter += 1
            #         else:
            #             break

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * vectors.size(0)

            # Print loss every 100 batches
            if i % 500 == 0:
                print(f'Epoch {epoch + 1}/{n_epochs}, Step {i}/{len(data_loader)}, Loss: {loss.item()}')
        train_losses.append(train_loss / len(data_loader.dataset))


        # Calculate average losses
        train_loss = train_loss / len(data_loader.dataset)

        # Record accuracy with scikit-learn

        # Print training statistics
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

        # Validate on the dev set
        model.eval()
        dev_loss = 0.0
        for i, (vectors, labels) in enumerate(dev_loader):
            vectors = vectors.to(device)
            labels = labels.to(device)

            outputs = model(vectors)
            loss = criterion(outputs, labels)
            dev_loss += loss.item() * vectors.size(0)
            # if i == 0:
            #     dev_preds = torch.sigmoid(outputs).cpu().detach().numpy()
            #     dev_labels = labels.cpu().detach().numpy().astype(int)
            dev_preds = np.append(dev_preds, torch.sigmoid(outputs).cpu().detach().numpy())
            dev_labels = np.append(dev_labels, labels.cpu().detach().numpy())
        print(len(dev_preds))
        print(len(dev_labels))

        mention_count = 0
        top_n_count = 0
        correct_count = 0
        current_correct_prob = 0.0
        entity_example_count = 0
        for i in range(len(dev_preds)):
            correct_flag = True
            if dev_labels[i] == 1:
                entity_example_count = 1
                mention_count += 1
                current_correct_prob = dev_preds[i]
            else:
                entity_example_count += 1
                if dev_preds[i] > current_correct_prob:
                    correct_flag = False
            if i == len(dev_preds) - 1 or dev_labels[i+1] == 1:
                if entity_example_count == 4:
                    top_n_count += 1
                if correct_flag:
                    correct_count += 1

        print(f'Correct count: {correct_count}/{mention_count} = {correct_count/mention_count*100}%')
        print(f"Realistic Top N count: {correct_count}/{top_n_count} = {correct_count/top_n_count*100}%")
        dev_loss = dev_loss / len(dev_loader.dataset)
        dev_preds = torch.from_numpy(dev_preds)
        dev_labels = torch.from_numpy(dev_labels).int()
        threshold = 0.7
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
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), 'models/xencoder/x_encoder_model.pt')

    # plt.figure()
    # plt.plot(range(n_epochs), train_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.title('Training Loss over Time')
    # plt.savefig('data/processed/x_encoder_training_loss.png')
