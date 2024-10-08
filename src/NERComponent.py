# Import external libraries
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import project code
from configs.load_configs import *


class Span:
    """
    Data object class containing information of a particular mention span.
    """
    def __init__(self, start: int, end: int, text: str):
        self.start = start
        self.end = end
        self.text = text


def bio_tags_to_spans(tokens: list[str], bio_tags: list[str]) -> list[Span]:
    """
    Given a list of tokens and their corresponding BIO annotations, return all mentions.
    """
    spans = []
    start_idx = None
    current_entity = None

    for idx, tag in enumerate(bio_tags):
        if tag.startswith('B-'):
            if current_entity is not None:
                spans.append(Span(start_idx, idx, " ".join(tokens[start_idx:idx])))
            current_entity = tag[2:]
            start_idx = idx
        elif tag.startswith('I-'):
            pass
            # if current_entity is None:
            #     start_idx = idx
            #     current_entity = tag[2:]
            # elif current_entity != tag[2:]:
            #     spans.append(Span(start_idx, idx, " ".join(tokens[start_idx:idx])))
            #     current_entity = tag[2:]
            #     start_idx = idx
        else:
            if current_entity is not None:
                spans.append(Span(start_idx, idx, " ".join(tokens[start_idx:idx])))
                current_entity = None
                start_idx = None

    if current_entity is not None:
        spans.append(Span(start_idx, len(bio_tags), " ".join(tokens[start_idx:len(bio_tags)])))

    return spans


class NERResult:
    """
    Data object class holding information of a sequence of text and its detected mentions.
    """
    def __init__(self, text: str,  tokens: list[str], spans: list[Span]):
        self.text = text
        self.tokens = tokens
        self.spans = spans


class NERComponent:
    """
    Re-implemented mention detector of MedLinker.
    """
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

    def predict(self, texts: str) -> NERResult:
        """
        Detect mentions given a piece of text.
        :param texts: Input piece of text.
        :return: A data object containing the original text and the detected mentions.
        """
        tokenized_sentence = []
        for word in texts.split():
            tokenized_word = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        maxlen = 128
        if len(tokenized_sentence) > maxlen:
            tokenized_sentence = tokenized_sentence[:maxlen]
        else:
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(maxlen - len(tokenized_sentence))]

        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        mask = torch.tensor(attn_mask, dtype=torch.long).unsqueeze(0).to(self.device)

        # Pass processed data through model.
        with torch.no_grad():
            outputs = self.model(ids, mask)
        logits = outputs.logits

        # Retrieve raw outputs from the model, interpret it and contruct the mention predictions.
        probabilities = torch.softmax(logits, dim=2).cpu()

        predictions = torch.argmax(probabilities, dim=2).cpu().numpy()
        texts_tokenized = [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in ids]
        tokens_processed = []
        sentence_prediction_processed = []
        for j in range(len(texts_tokenized[0])):
            if texts_tokenized[0][j].startswith("##"):
                tokens_processed[-1] += texts_tokenized[0][j][2:]
            elif texts_tokenized[0][j] == "[CLS]" \
                    or texts_tokenized[0][j] == "[SEP]" \
                    or texts_tokenized[0][j] == "[PAD]":
                continue
            else:
                tokens_processed.append(texts_tokenized[0][j])
                sentence_prediction_processed.append(self.id2label[predictions[0][j]])
        result = NERResult(texts,
                           tokens_processed,
                           bio_tags_to_spans(tokens_processed, sentence_prediction_processed))
        return result


if __name__ == "__main__":
    ner_component = NERComponent()
    sentence = "The patient was prescribed 100mg of ibuprofen for pain relief."
    ner_result = ner_component.predict(sentence)
    print(ner_result.text)
    print(ner_result.tokens)
    for span in ner_result.spans:
        print(f"Entity: {span.text} (start: {span.start}, end: {span.end})")
    print()
