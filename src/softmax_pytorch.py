import torch.nn as nn


class SoftmaxClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(768, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x
