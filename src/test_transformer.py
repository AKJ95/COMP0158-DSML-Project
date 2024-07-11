from transformers import AutoModel, AutoTokenizer
import torch
from pytorch_transformers import BertModel, BertTokenizer

if __name__ == '__main__':
    return_tokens = True
    tokens = ["hello", "world", "this", "is", "a", "test"]
    pytt_model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
    pytt_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # pytt_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
    # pytt_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoding_map = [pytt_tokenizer.encode(t) for t in tokens]
    sent_encodings = sum(encoding_map, [])
    sent_encodings = pytt_tokenizer.encode(pytt_tokenizer.cls_token) + \
                     sent_encodings + \
                     pytt_tokenizer.encode(pytt_tokenizer.sep_token)
    input_ids = torch.tensor([sent_encodings])
    all_hidden_states, all_attentions = pytt_model(input_ids)[-2:]
    all_hidden_states = sum([all_hidden_states[i] for i in [-1, -2, -3, -4]])
    all_hidden_states = all_hidden_states[0]  # batch size 1
    all_hidden_states = all_hidden_states[1:-1]

    # align and merge subword embeddings (average)
    tok_embeddings = []
    encoding_idx = 0
    for tok, tok_encodings in zip(tokens, encoding_map):
        tok_embedding = torch.zeros(pytt_model.config.hidden_size)
        for tok_encoding in tok_encodings:
            tok_embedding += all_hidden_states[encoding_idx]
            encoding_idx += 1
        tok_embedding = tok_embedding / len(tok_encodings)  # avg of subword embs
        tok_embedding = tok_embedding.detach().cpu().numpy()

        if return_tokens:
            tok_embeddings.append((tok, tok_embedding))
        else:
            tok_embeddings.append(tok_embedding)

    print(tok_embeddings)
