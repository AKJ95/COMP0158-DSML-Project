# Import external libraries
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import project code
from configs.load_configs import *


class NERComponent:
    def __init__(self):
        ner_configs = get_ner_training_config()
        self.tokenizer_root = ner_configs.tokenizer_path
        self.model_path = ner_configs.model_path
        self.num_labels = len(ner_configs.label2id)
        self.label2id = ner_configs.label2id
        self.id2label = ner_configs.id2label
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_root)
        self.model = AutoModelForTokenClassification.from_pretrained(ner_configs.ner_model_name,
                                                                     num_labels=self.num_labels,
                                                                     id2label=self.id2label,
                                                                     label2id=self.label2id
                                                                     )
        self.model.load_state_dict(torch.load(self.model_path))

    def predict(self, text: str):
        self.model.eval()


if __name__ == "__main__":
    ner_component = NERComponent()
    print(ner_component.tokenizer)
    print(ner_component.model)
