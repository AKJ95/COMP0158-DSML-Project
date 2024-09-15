# Code from [1] is consulted and adopted.

import logging

import torch as th
from pytorch_transformers import BertModel, BertTokenizer

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

PYTT_CONFIG = {'external': True, 'lower_case': True, 'name': 'scibert_scivocab_uncased',
               'path': 'models/BERT/scibert_scivocab_uncased'}

if PYTT_CONFIG['external']:
    pytt_tokenizer = BertTokenizer.from_pretrained(PYTT_CONFIG['path'], do_lower_case=PYTT_CONFIG['lower_case'])
    pytt_model = BertModel.from_pretrained(PYTT_CONFIG['path'], output_hidden_states=True, output_attentions=True)
else:
    pytt_tokenizer = BertTokenizer.from_pretrained(PYTT_CONFIG['name'], do_lower_case=PYTT_CONFIG['lower_case'])
    pytt_model = BertModel.from_pretrained(PYTT_CONFIG['name'], output_hidden_states=True, output_attentions=True)

device = 'cuda' if th.cuda.is_available() else 'cpu'
pytt_model.eval()
pytt_model.to(device)


def get_num_features(tokens):
    return len(sum([pytt_tokenizer.encode(t) for t in tokens], [])) + 2  # plus CLS and SEP


def toks2vecs(tokens, layers=None, subword_op='avg', layer_op='sum', return_tokens=True):
    if layers is None:
        layers = [-1, -2, -3, -4]
    encoding_map = [pytt_tokenizer.encode(t) for t in tokens]
    sent_encodings = sum(encoding_map, [])
    sent_encodings = pytt_tokenizer.encode(pytt_tokenizer.cls_token) + \
                     sent_encodings + \
                     pytt_tokenizer.encode(pytt_tokenizer.sep_token)

    input_ids = th.tensor([sent_encodings]).to(device)
    all_hidden_states, all_attentions = pytt_model(input_ids)[-2:]

    all_hidden_states = sum([all_hidden_states[i] for i in layers])
    all_hidden_states = all_hidden_states[0]  # batch size 1
    all_hidden_states = all_hidden_states[1:-1]  # ignoring CLS and SEP

    # align and merge subword embeddings (average)
    tok_embeddings = []
    encoding_idx = 0
    for tok, tok_encodings in zip(tokens, encoding_map):
        tok_embedding = th.zeros(pytt_model.config.hidden_size).to(device)
        for tok_encoding in tok_encodings:
            tok_embedding += all_hidden_states[encoding_idx]
            encoding_idx += 1
        tok_embedding = tok_embedding / len(tok_encodings)  # avg of subword embs
        tok_embedding = tok_embedding.detach().cpu().numpy()

        if return_tokens:
            tok_embeddings.append((tok, tok_embedding))
        else:
            tok_embeddings.append(tok_embedding)

    return tok_embeddings


if __name__ == '__main__':
    sent = "Hello World !"
    sent_embeddings = toks2vecs(sent.split())
