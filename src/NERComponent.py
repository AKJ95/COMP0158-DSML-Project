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
        self.model.eval()

    def predict(self, texts: list[str]):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        print(inputs)



if __name__ == "__main__":
    ner_component = NERComponent()
    sentences = ["The patient was prescribed 100mg of ibuprofen for pain relief.",
                 "The patient was prescribed 500mg of amoxicillin for infection."]
    print(ner_component.predict(sentences))
