# Import external libraries
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import project code
from configs.load_configs import *


class NERComponent:
    def __init__(self):
        ner_configs = get_ner_training_config()
        self.tokenizer_root = ner_configs.tokenizer_path
        self.model_path = ner_configs.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_root)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)

    def predict(self, text: str):
        pass


if __name__ == "__main__":
    ner_component = NERComponent()
    print(ner_component.tokenizer)
    print(ner_component.model)
