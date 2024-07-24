from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import torch
import torch.nn as nn


class SoftMax_CLF(nn.Module):

    def __init__(self, n_classes):
        super(SoftMax_CLF, self).__init__()
        self.fc = nn.Linear(768, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    model = Sequential([
        Dense(18426, activation='softmax', input_shape=(768,)),
    ])
    model.load_weights("models/Classifiers/softmax.cui.h5")
    tf_weights = model.weights[0].numpy()
    tf_biases = model.weights[1].numpy()
    pytorch_softmax = SoftMax_CLF(18426)
    pytorch_softmax.load_state_dict({
        'fc.weight': torch.from_numpy(np.transpose(tf_weights)),
        'fc.bias': torch.from_numpy(tf_biases),
    })

    toy_input = torch.rand((64, 768))
    toy_output = pytorch_softmax(toy_input)
    print(toy_output.shape)
    print(toy_output)
