# Code from [1] is consulted and adopted.
import json
import logging

import spacy

from NERComponent import NERComponent
from medlinker import MedLinker
from pytt_hf_custom_tokenizer import toks2vecs
from umls import umls_kb_full_2017 as umls_kb
from train_x_encoder import MLP
import torch


sci_nlp = spacy.load('en_core_sci_md', disable=['tagger', 'parser', 'ner'])

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logging.info('Loading MedLinker ...')


ngram_db_path = 'data/processed/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
ngram_map_path = 'data/processed/umls.2017AA.active.st21pv.aliases.5toks.map'
cui_vsm_path = 'data/processed/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'


# Record gold labels and their corresponding predictions.
gold_labels = []
pred_labels = []

# Load MedLinker and set up its configuration.
# This pipeline is strictly used to evaluate MedLinker with the cross-encoder re-ranker.
# As such, all three constituent predictors should be loaded to replicated the result in the thesis.
print('Loading MedNER ...')
medner = NERComponent()

print('Loading MedLinker ...')
medlinker = MedLinker(medner, umls_kb)
medlinker.load_string_matcher(ngram_db_path, ngram_map_path)
medlinker.load_cui_softmax_pt()
medlinker.load_cui_VSM(cui_vsm_path)

# Load the trained cross-encoder re-ranker.
model = MLP()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load('models/xencoder/x_encoder_model_ens.pt'))
model.eval()


def read_mm_converted(mm_set_path):
    """
    Read the MedMentions dataset from a json file.
    :param mm_set_path: filepath to the MedMentions dataset
    :return: Content of the MedMentions dataset.
    """
    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])


if __name__ == '__main__':
    perf_stats = {'n_gold_spans': 0, 'n_pred_spans': 0, 'n_sents': 0, 'n_docs': 0}   # Obsolete
    perf_cui = {'tp': 0, 'fp': 0, 'fn': 0}

    # Load MedMentions test split.
    logging.info('Loading MedMentions ...')
    mm_docs = read_mm_converted('data/processed/mm_converted.test.json')

    # Keep track of key statistics
    logging.info('Processing Instances ...')
    span_count = 0   # Number of predictions
    gold_span_count = 0  # Number of gold spans
    in_top_n_count = 0  # Number of instances where MedLinker correctly put the gold entity in the top n predictions
    skip_count = 0   # Obsolete
    x_encoder_example_count = 0  # Obsolete
    x_encoder_skipped_count = 0  # Obsolete
    vectors = []    # Obsolete
    labels = []     # Obsolete
    correct_count = 0   # True positive count

    # Iterate through all documents in MedMentions test split.
    for doc_idx, doc in enumerate(mm_docs):
        perf_stats['n_docs'] += 1    # Obsolete

        logging.info('At doc #%d' % doc_idx)

        # Record gold entities in all sentences of the document.
        gold_ents = set()
        for gold_sent in doc['sentences']:
            for gold_span in gold_sent['spans']:
                gold_ents.add(gold_span['cui'].lstrip('UMLS:'))

        pred_ents = set()
        # Iterate through all sentences in the document.
        for gold_sent in doc['sentences']:
            # Predict for all gold spans by retrieving the top 4 predictions made by the ensemble.
            gold_spans = [(span['start'], span['end']) for span in gold_sent['spans']]
            sent_preds = medlinker.predict(' '.join(gold_sent['tokens']),
                                           gold_tokens=gold_sent['tokens'],
                                           gold_spans=gold_spans,
                                           top_n=4)

            pred_reranked_entities = []

            # Iterate through all spans again to make final prediction
            for i in range(len(sent_preds['spans'])):
                # Construct mention tokens for the current span
                embedding_tokens = []
                embedding_tokens.extend(gold_sent['tokens'][:sent_preds['spans'][i]['start']])
                embedding_tokens.append('[M_s]')
                embedding_tokens.extend(gold_sent['tokens'][sent_preds['spans'][i]['start']:sent_preds['spans'][i]['end']])
                embedding_tokens.append('[M_e]')
                embedding_tokens.extend(gold_sent['tokens'][sent_preds['spans'][i]['end']:])
                embedding_tokens.append('[SEP]')

                span_count += 1
                max_score = 0.0
                max_entity = None

                # Iterate through the top 4 predictions of the current span.
                for j in range(min(4, len(sent_preds['spans'][i]['cui']))):

                    # Retrieve name and definition (if available) of the predicted entity from UMLS database
                    pred_entity_kb = umls_kb.get_entity_by_cui(sent_preds['spans'][i]['cui'][j][0])
                    pred_entity_name = pred_entity_kb['Name'] if pred_entity_kb else ''

                    # Construct entity tokens as specified in the thesis
                    pred_entity_name_tokens = [t.text.lower() for t in sci_nlp(pred_entity_name)]
                    pred_entity_tokens = []
                    if pred_entity_kb and pred_entity_kb['DEF']:
                        pred_entity_def = pred_entity_kb['DEF'][0]
                        pred_entity_def_tokens = [t.text.lower() for t in sci_nlp(pred_entity_def)]
                        pred_entity_tokens = pred_entity_name_tokens + ['[ENT]'] + pred_entity_def_tokens
                    else:
                        x_encoder_skipped_count += 1
                        pred_entity_tokens = pred_entity_name_tokens + ['[ENT]'] + pred_entity_name_tokens

                    # Construct combined mention-entity tokens to be passed into the cross-encoder as specified in the
                    # thesis. And encode it.
                    toy_vec = toks2vecs((embedding_tokens + pred_entity_tokens)[:128])
                    toy_vec = torch.from_numpy(toy_vec).float().unsqueeze(0)
                    toy_vec = toy_vec.to(device)

                    # Score the mention-entity pair with the MLP of the re-ranker.
                    pred = model(toy_vec)
                    score = torch.sigmoid(pred).item()
                    # Keep track of the top-scoring entity
                    if score > max_score:
                        max_score = score
                        max_entity = sent_preds['spans'][i]['cui'][j][0]

                pred_entities = [entry[0] for entry in sent_preds['spans'][i]['cui']]
                gold_entity = gold_sent['spans'][i]['cui'].lstrip('UMLS:')
                pred_reranked_entities.append(max_entity)

                # Check if the top-scoring entity by the re-ranker is the correct entity.
                if gold_entity == max_entity:
                    correct_count += 1

                # Check if the correct entity exists in the top 4 predictions.
                if gold_entity in pred_entities:
                    in_top_n_count += 1

            for pred_span in sent_preds['spans']:
                for pred_cui in pred_span['cui']:
                    pred_ents.add(pred_cui[0])

            gold_span_count += len(gold_spans)

            assert len(pred_reranked_entities) == len(sent_preds['spans'])

            # Add gold labels and their corresponding prediction to gold_labels and pred_labels for record.
            for gold_span in gold_sent['spans']:
                gold_span_start = gold_span['start']
                gold_span_end = gold_span['end']
                gold_span_cui = gold_span['cui'].lstrip('UMLS:')
                found_pred = False
                gold_labels.append(gold_span_cui)
                for i in range(len(sent_preds['spans'])):
                    pred_span_start = sent_preds['spans'][i]['start']
                    pred_span_end = sent_preds['spans'][i]['end']
                    pred_span_cui = pred_reranked_entities[i]
                    if gold_span_start == pred_span_start and gold_span_end == pred_span_end:
                        pred_labels.append(pred_span_cui)
                        found_pred = True
                        break
                if not found_pred:
                    pred_labels.append(None)

        # Calculate span-level metrics
        precision = correct_count / span_count * 100
        recall = correct_count / (correct_count + gold_span_count - span_count) * 100
        f1 = 2 * ((precision * recall) / (precision + recall))
        tp = correct_count
        fp = span_count - correct_count
        fn = gold_span_count - span_count

        # Print metrics readings.
        print(f"[CUI]\tP:{precision:.2f}\tR:{recall:.2f}\tF1:{f1:.2f}\tTP:{tp}\tFP:{fp}\tFN:{fn}")
        print(f"Span-level top k recall: {in_top_n_count}/{gold_span_count} ({in_top_n_count / gold_span_count * 100:.2f}%)")

    # Save gold labels and their corresponding predictions
    print("Saving results...")
    if gold_labels and pred_labels:
        # Convert the list to a JSON string
        results = {"gold_labels": gold_labels, "pred_labels": pred_labels}
        results_str = json.dumps(results)
        # Write the string to a file
        with open('results/ens_rerank.txt', 'w') as file:
            file.write(results_str)
