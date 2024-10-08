import json
import logging

import numpy as np
import spacy

from NERComponent import NERComponent
from medlinker import MedLinker
from pytt_hf_custom_tokenizer import toks2vecs
from umls import umls_kb_full_2017 as umls_kb

sci_nlp = spacy.load('en_core_sci_md', disable=['tagger', 'parser', 'ner'])

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logging.info('Loading MedLinker ...')


ngram_db_path = 'data/processed/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
ngram_map_path = 'data/processed/umls.2017AA.active.st21pv.aliases.5toks.map'
cui_vsm_path = 'data/processed/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'


print('Loading MedNER ...')
medner = NERComponent()

print('Loading MedLinker ...')
medlinker = MedLinker(medner, umls_kb)
medlinker.load_string_matcher(ngram_db_path, ngram_map_path)
medlinker.load_cui_softmax_pt()
medlinker.load_cui_VSM(cui_vsm_path)


def read_mm_converted(mm_set_path):
    """
    Read in MedMentions dataset.
    """
    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])


# def calc_p(metrics):
#     try:
#         return metrics['tp'] / (metrics['tp'] + metrics['fp'])
#     except ZeroDivisionError:
#         return 0
#
#
# def calc_r(metrics):
#     try:
#         return metrics['tp'] / (metrics['tp'] + metrics['fn'])
#     except ZeroDivisionError:
#         return 0
#
#
# def calc_f1(metrics):
#     try:
#         p = calc_p(metrics)
#         r = calc_r(metrics)
#         return 2 * ((p * r) / (p + r))
#     except ZeroDivisionError:
#         return 0
#
#
# def calc_acc(metrics):
#     try:
#         # return metrics['tp'] / sum(metrics.values())
#         return metrics['tp'] / (metrics['tp'] + metrics['fp'] + metrics['fn'])
#     except ZeroDivisionError:
#         return 0
#
#
# def calc_counts(metrics):
#     metrics['n'] = sum(metrics.values())
#     return metrics
#
#
# def stringify_metrics(metrics):
#     metrics_counts = calc_counts(metrics)
#     return ' '.join(['%s:%d' % (l.upper(), c) for l, c in metrics_counts.items()])


if __name__ == '__main__':
    perf_stats = {'n_gold_spans': 0, 'n_pred_spans': 0, 'n_sents': 0, 'n_docs': 0}
    perf_cui = {'tp': 0, 'fp': 0, 'fn': 0}

    # Load in MedMentions training set
    logging.info('Loading MedMentions ...')
    mm_docs = read_mm_converted('data/processed/mm_converted.dev.json')

    logging.info('Processing Instances ...')
    span_count = 0
    in_top_n_count = 0
    skip_count = 0
    x_encoder_example_count = 0
    x_encoder_skipped_count = 0
    vectors = []
    labels = []
    correct_count = 0

    # Iterate through MedMentions training set
    for doc_idx, doc in enumerate(mm_docs):
        perf_stats['n_docs'] += 1

        logging.info('At doc #%d' % doc_idx)

        gold_ents = set()
        for gold_sent in doc['sentences']:
            for gold_span in gold_sent['spans']:
                gold_ents.add(gold_span['cui'].lstrip('UMLS:'))

        pred_ents = set()

        # Iterate through each sentence
        for gold_sent in doc['sentences']:

            # Get MedLinker's top 4 entity predictions for each gold span.
            gold_spans = [(span['start'], span['end']) for span in gold_sent['spans']]
            sent_preds = medlinker.predict(' '.join(gold_sent['tokens']),
                                           gold_tokens=gold_sent['tokens'],
                                           gold_spans=gold_spans,
                                           top_n=4)

            # Iterate through each gold span now:
            for i in range(len(sent_preds['spans'])):
                # Collect the tokens that form the sentence containing the mention, and convert it into the format
                # described in the thesis before being passed into the cross-encoder.
                embedding_tokens = []
                embedding_tokens.extend(gold_sent['tokens'][:sent_preds['spans'][i]['start']])
                embedding_tokens.append('[M_s]')
                embedding_tokens.extend(gold_sent['tokens'][sent_preds['spans'][i]['start']:sent_preds['spans'][i]['end']])
                embedding_tokens.append('[M_e]')
                embedding_tokens.extend(gold_sent['tokens'][sent_preds['spans'][i]['end']:])
                embedding_tokens.append('[SEP]')

                # Retrieve the name and the definition of the gold entity and organise the tokens that form them into
                # the format as described in the paper before being passed into the cross-encoder.
                gold_entity_cui = gold_sent['spans'][i]['cui'].lstrip('UMLS:')
                gold_entity_kb = umls_kb.get_entity_by_cui(gold_sent['spans'][i]['cui'].lstrip('UMLS:'))
                gold_entity_name = gold_entity_kb['Name'] if gold_entity_kb else ' '.join(gold_sent['tokens'][sent_preds['spans'][i]['start']:sent_preds['spans'][i]['end']])
                if gold_entity_kb and gold_entity_kb['DEF']:
                    gold_entity_def = gold_entity_kb['DEF'][0]
                else:
                    gold_entity_def = gold_entity_name
                    skip_count += 1
                    x_encoder_skipped_count += 1

                # Concatenate the mention tokens and the gold entity definition tokens and pass it through the
                # the cross-encoder. This forms a positive mention-entity pair. A label of 1 is duly assigned to it.
                gold_entity_tokens = [t.text.lower() for t in sci_nlp(gold_entity_name)] + ['[ENT]'] + [t.text.lower()
                                                                                                        for t in
                                                                                                        sci_nlp(
                                                                                                            gold_entity_def)]

                # print(gold_entity_name)
                # print(gold_sent['spans'][i]['cui'].lstrip('UMLS:'))
                # print(gold_entity_tokens)
                # print("This is gold embedding")
                # print(embedding_tokens + gold_entity_tokens)
                gold_toy_vec = toks2vecs((embedding_tokens + gold_entity_tokens)[:128])
                vectors.append(gold_toy_vec)
                labels.append(1)
                # print(gold_toy_vec[:10])

                span_count += 1
                x_encoder_example_count += 1

                # Iterate through the top 4 predictions made by MedLinker for each span.
                for j in range(min(4, len(sent_preds['spans'][i]['cui']))):
                    pred_entity_kb = umls_kb.get_entity_by_cui(sent_preds['spans'][i]['cui'][j][0])
                    pred_entity_name = pred_entity_kb['Name'] if pred_entity_kb else ''
                    # If the predicted entity is the gold entity, skip this example.
                    if j == 0 and pred_entity_name == gold_entity_name:
                        correct_count += 1
                    # If the predicted entity is different to the gold entity, then it forms a negative mention-entity
                    # pair. Create the negative training example in the same way as above, by forming the mention
                    # and entity definition tokens and pass them through the cross-encoder.
                    if pred_entity_name != gold_entity_name:
                        x_encoder_example_count += 1
                        pred_entity_name_tokens = [t.text.lower() for t in sci_nlp(pred_entity_name)]
                        pred_entity_tokens = []
                        if pred_entity_kb and pred_entity_kb['DEF']:
                            pred_entity_def = pred_entity_kb['DEF'][0]
                            pred_entity_def_tokens = [t.text.lower() for t in sci_nlp(pred_entity_def)]
                            pred_entity_tokens = pred_entity_name_tokens + ['[ENT]'] + pred_entity_def_tokens
                        else:
                            x_encoder_skipped_count += 1
                            pred_entity_tokens = pred_entity_name_tokens + ['[ENT]'] + pred_entity_name_tokens
                        # print(pred_entity_tokens)
                        # print("This is counterexample embeddings")
                        # print(embedding_tokens + pred_entity_tokens)
                        toy_vec = toks2vecs((embedding_tokens + pred_entity_tokens)[:128])
                        # print(toy_vec[:10])
                        vectors.append(toy_vec)
                        labels.append(0)

                pred_entities = [entry[0] for entry in sent_preds['spans'][i]['cui']]
                gold_entity = gold_sent['spans'][i]['cui'].lstrip('UMLS:')
                if gold_entity in pred_entities:
                    in_top_n_count += 1
            for pred_span in sent_preds['spans']:
                for pred_cui in pred_span['cui']:
                    pred_ents.add(pred_cui[0])

        # Print some in-time statistics showing the progress of encoding the examples.
        perf_cui['tp'] += len(gold_ents.intersection(pred_ents))
        perf_cui['fp'] += len([pred_ent for pred_ent in pred_ents if pred_ent not in gold_ents])
        perf_cui['fn'] += len([gold_ent for gold_ent in gold_ents if gold_ent not in pred_ents])

        # in-progress performance metrics
        # p = calc_p(perf_cui) * 100
        # r = calc_r(perf_cui) * 100
        # f = calc_f1(perf_cui) * 100
        # a = calc_acc(perf_cui) * 100

        # counts = calc_counts(perf_cui)
        # counts_str = '\t'.join(['%s:%d' % (l.upper(), c) for l, c in counts.items()])
        print(f"Percentage of correct: {correct_count}/{span_count} ({correct_count / span_count * 100:.2f}%)")
        print(f"Top k Recall per span: {in_top_n_count}/{span_count} ({in_top_n_count / span_count * 100:.2f}%)")
        print(f"Examples without official definitions: {x_encoder_skipped_count}/{x_encoder_example_count} ({x_encoder_skipped_count / x_encoder_example_count * 100:.2f}%)")

        # if doc_idx >= 100:
        #     break

    # Store the vectors and labels
    vector_np = np.vstack(vectors)
    labels_np = np.array(labels)
    # print(vector_np.shape)
    # print(labels_np.shape)
    # np.save('data/processed/x_encoder_vectors_ens_test.npy', vector_np)
    # np.save('data/processed/x_encoder_labels_ens_test.npy', labels_np)
