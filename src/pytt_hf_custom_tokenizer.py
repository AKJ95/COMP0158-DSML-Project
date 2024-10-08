# Code from [1] is consulted and adopted.

import logging

import torch as th
from pytorch_transformers import BertModel, BertTokenizer

# Loads the SciBert model and tokenizer.
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%d-%b-%y %H:%M:%S')

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
pytt_tokenizer.add_special_tokens({'additional_special_tokens': ['[M_s]', '[M_e]', '[ENT]']})
# print(pytt_tokenizer.all_special_tokens)
toy_default_tokens = ["Hello", "World", "!"]
toy_tokens = ["[M_s]", "Hello", "World", "!", "[M_e]"]
output_1 = [pytt_tokenizer.encode(toy_token) for toy_token in toy_tokens]
output_2 = [pytt_tokenizer.encode(toy_default_token) for toy_default_token in toy_default_tokens]
# print("output_1:", output_1)
# print("output_2:", output_2)


def get_num_features(tokens):
    return len(sum([pytt_tokenizer.encode(t) for t in tokens], [])) + 2  # plus CLS and SEP


def toks2vecs(tokens, layers=None, subword_op='avg', layer_op='sum', return_tokens=True):
    """
    Returns the contextual embeddings of the input sentence represented by tokens by taking the output of the final
    hidden layer of the beginning [CLS] token.
    """
    if layers is None:
        layers = [-1]
    encoding_map = [pytt_tokenizer.encode(t) for t in tokens]
    sent_encodings = sum(encoding_map, [])
    sent_encodings = pytt_tokenizer.encode(pytt_tokenizer.cls_token) + \
                     sent_encodings + \
                     pytt_tokenizer.encode(pytt_tokenizer.sep_token)

    input_ids = th.tensor([sent_encodings]).to(device)
    output = pytt_model(input_ids)
    all_hidden_states, all_attentions = output[-2:]

    # print(len(all_hidden_states))
    all_hidden_states = sum([all_hidden_states[i] for i in layers])
    # print(len(all_hidden_states))
    all_hidden_states = all_hidden_states[0]  # batch size 1
    # print(len(all_hidden_states))
    cls_hidden_states = all_hidden_states[0]

    return cls_hidden_states.detach().cpu().numpy()


if __name__ == '__main__':
    sent = "Hello World !"
    sent_embeddings = toks2vecs(toy_tokens)
    print(sent_embeddings)
