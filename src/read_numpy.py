import numpy as np


train_labels = np.load('data/processed/x_encoder_labels_2017.npy')
positive_counts = np.sum(train_labels)
positive_proportion = positive_counts / len(train_labels)
weight_positive = 1 / positive_proportion
print(f"Total number of labels: {len(train_labels)}")
print(f"Positive labels: {positive_counts}")
print(f"Positive proportion: {positive_proportion}")
print(f"Weight for positive class: {weight_positive}")
