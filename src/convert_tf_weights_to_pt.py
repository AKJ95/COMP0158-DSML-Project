from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import torch

from configs.load_configs import *
from softmax_pytorch import SoftmaxClassifier


if __name__ == '__main__':
    configs = get_softmax_classifier_configuration()
    model = Sequential([
        Dense(18426, activation='softmax', input_shape=(768,)),
    ])
    model.load_weights(configs.softmax_tf_path)
    tf_weights = model.weights[0].numpy()
    tf_biases = model.weights[1].numpy()
    pytorch_softmax_model = SoftmaxClassifier(18426)
    pytorch_softmax_model.load_state_dict({
        'fc.weight': torch.from_numpy(np.transpose(tf_weights)),
        'fc.bias': torch.from_numpy(tf_biases),
    })
    torch.save(pytorch_softmax_model.state_dict(), configs.softmax_pt_path)
