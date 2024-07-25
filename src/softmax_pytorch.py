import joblib

import torch.nn as nn


class SoftmaxClassifier(nn.Module):

    def __init__(self, n_classes: int, label_mapping_path: str):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(768, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.label_mapping = joblib.load(label_mapping_path)
        self.label_mapping = {i: l for l, i in self.label_mapping.items()}

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        x = self.forward(x)[0].cpu().detach().numpy()
        score_dict = {}
        for i in range(len(x)):
            cui_label = self.label_mapping[i]
            cui_label = cui_label.lstrip('UMLS:')
            score_dict[cui_label] = x[i]
        return score_dict
