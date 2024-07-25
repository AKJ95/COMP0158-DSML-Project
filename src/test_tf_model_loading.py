import joblib

import numpy as np
import torch
import torch.nn as nn

from softmax_pytorch import SoftmaxClassifier


def load_precomputed_embeddings(precomputed_path, mm_ann, label_mapping=None):
    all_anns, all_vecs = [], []

    with open(precomputed_path, 'r') as f:
        for line in f:
            elems = line.split('\t')
            cui = elems[3]
            sty = elems[4]
            vec = np.array(list(map(float, elems[-1].split())), dtype=np.float32)

            if mm_ann == 'sty':
                all_anns.append(sty)
            elif mm_ann == 'cui':
                all_anns.append(cui)

            all_vecs.append(vec)

    if label_mapping is None:
        label_mapping = {a: i + 1 for i, a in enumerate(set(all_anns))}
        label_mapping['UNK'] = 0

    X = np.vstack(all_vecs)
    y = []
    for ann in all_anns:
        try:
            y.append(label_mapping[ann])
        except KeyError:
            y.append(0)

    return X, y, label_mapping


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mm_ann = 'cui'
    path_precomputed_train_vecs = 'data/processed/mm_st21pv.train.scibert_scivocab_uncased.precomputed'
    path_precomputed_dev_vecs = 'data/processed/mm_st21pv.dev.scibert_scivocab_uncased.precomputed'
    path_precomputed_dev_map = 'models/Classifiers/softmax.cui.map'

    print('Loading precomputed ...')
    mapping = label_mapping = joblib.load(path_precomputed_dev_map)
    mapping = {i: l for l, i in mapping.items()}
    X_train, y_train, train_label_mapping = load_precomputed_embeddings(path_precomputed_train_vecs, mm_ann, mapping)
    X_dev, y_dev, _ = load_precomputed_embeddings(path_precomputed_dev_vecs, mm_ann, train_label_mapping)
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(np.array(y_train)).to(device)
    X_dev = torch.from_numpy(X_dev).to(device)
    y_dev = torch.from_numpy(np.array(y_dev)).to(device)

    pytorch_softmax = SoftmaxClassifier(18426, path_precomputed_dev_map)
    pytorch_softmax.load_state_dict(torch.load('models/Classifiers/softmax.cui.pt'))
    pytorch_softmax.to(device)
    pytorch_softmax.eval()

    for training_id in range(16):
        score_dict = pytorch_softmax.predict(X_train[training_id:training_id + 1])
        max_label = None
        max_score = 0
        for k, v in score_dict.items():
            if v > max_score:
                max_score = v
                max_label = k
        print(f"Prediction: {max_label} with score {max_score}")
