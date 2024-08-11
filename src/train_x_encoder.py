import numpy as np


if __name__ == '__main__':
    x_encoder_vectors = np.load('data/processed/x_encoder_vectors.npy')
    x_encoder_labels = np.load('data/processed/x_encoder_labels.npy')
    print(x_encoder_vectors.shape)
    print(x_encoder_labels.shape)
    print(x_encoder_labels[:20])
