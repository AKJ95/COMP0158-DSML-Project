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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str]):
        # Tokenize and make predictions.
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(ids, mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).cpu().numpy()
        texts_tokenized = [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs["input_ids"]]
        texts_tokenized_processed = []
        predictions_processed = []
        for i in range(len(texts_tokenized)):
            tokens_processed = []
            sentence_prediction_processed = []
            for j in range(len(texts_tokenized[i])):
                if texts_tokenized[i][j].startswith("##"):
                    tokens_processed[-1] += texts_tokenized[i][j][2:]
                elif texts_tokenized[i][j] == "[CLS]" \
                        or texts_tokenized[i][j] == "[SEP]" \
                        or texts_tokenized[i][j] == "[PAD]":
                    continue
                else:
                    tokens_processed.append(texts_tokenized[i][j])
                    sentence_prediction_processed.append(self.id2label[predictions[i][j]])
            texts_tokenized_processed.append(tokens_processed)
            predictions_processed.append(sentence_prediction_processed)
        print(texts_tokenized_processed)
        return texts_tokenized_processed, predictions_processed


if __name__ == "__main__":
    ner_component = NERComponent()
    sentences = ["The patient was prescribed 100mg of ibuprofen for pain relief.",
                 "The patient was prescribed 500mg of amoxicillin for infection."]
    print(ner_component.predict(sentences))
