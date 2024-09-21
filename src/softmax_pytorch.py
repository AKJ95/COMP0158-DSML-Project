import joblib

import torch
import torch.nn as nn


class SoftmaxClassifier(nn.Module):
    """
    MLP classifier implemented with PyTorch for MedLinker
    """

    def __init__(self, n_classes: int, label_mapping_path: str, checkpoint_path=None):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(768, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.label_mapping = joblib.load(label_mapping_path)
        self.label_mapping = {i: l for l, i in self.label_mapping.items()}
        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path))
        self.eval()

    def forward(self, x):
        """
        Forward pass
        :param x: inputs (contextual embeddings of mentions)
        :return: output logits
        """
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def predict(self, x, threshold=0.5):
        """
        Predicts the CUI for a mention.
        :param x: inputs (contextual embeddings of mentions)
        :param threshold: If the softmax score is below this threshold, the CUI is not considered.
        :return: List of CUI predictions with their corresponding softmax scores.
        """
        x = self.forward(x).detach().cpu().numpy()[0]
        score_dict = {}
        for i in range(len(x)):
            if x[i] > threshold:
                cui_label = self.label_mapping[i]
                cui_label = cui_label.lstrip('UMLS:')
                score_dict[cui_label] = x[i]

        score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return score_dict
