# Code from [1] is consulted and adopted.

import json
import logging

import numpy as np

from pytt_hf import toks2vecs
from pytt_hf import PYTT_CONFIG

# Set up logging configuration.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def iterate_docs_converted(split_path):
    """
    Wrapper to iterate over converted MedMentions dataset.

    :param split_path: The split file path to the converted dataset.
    :return: A generator of documents in the specified split.
    """
    with open(split_path, 'r') as json_f:
        dataset = json.load(json_f)

    for doc in dataset['docs']:
        yield doc


def get_ctx_vec(sent_ctx_out, sent_tokens, span_idx_start, span_idx_end, normalize=False):
    """
    Compute the context vector for a span in a sentence.

    :param sent_ctx_out:
    :param sent_tokens: Tokens of the sentence.
    :param span_idx_start: Starting index of the span.
    :param span_idx_end: Ending index of the span.
    :param normalize: Whether to normalize the context vector.
    :return:
    """
    span_toks = sent_tokens[span_idx_start:span_idx_end]
    span_ctx_out = sent_ctx_out[span_idx_start:span_idx_end]

    span_ctx_toks = [t for t, v in span_ctx_out]

    # sanity check - prob. unnecessary ...
    assert span_ctx_toks == span_toks

    span_ctx_vecs = [v for t, v in span_ctx_out]
    span_ctx_vec = np.array(span_ctx_vecs).mean(axis=0)

    if normalize:
        span_ctx_vec = span_ctx_vec / np.linalg.norm(span_ctx_vec)

    return span_ctx_vec


if __name__ == '__main__':

    # Load MedMentions ST21pv training set.
    train_docs = list(iterate_docs_converted('data/processed/mm_converted.train.json'))

    skipped_anns = 0
    concept_vecs = {}
    st_ann_vecs = {}  # pooled over all annotations belonging to the same ST

    # Iterate through MedMentions ST21pv training set.
    for doc_idx, doc in enumerate(train_docs):

        logging.info('#Docs:%d #Concepts:%d #Types:%d #Skipped Ann.:%d' % (
        doc_idx, len(concept_vecs), len(st_ann_vecs), skipped_anns))

        # Iterate through each sentence
        for sent in doc['sentences']:

            # Get contextual embeddings for each token in the sentence.
            sent_ctx_out = toks2vecs(sent['tokens'])

            # Iterate through the gold mention spans in the sentence.
            for ent in sent['spans']:

                # Extract the entity label and the contextual embedding of the span.
                ent['cui'] = ent['cui'].lstrip('UMLS:')
                span_ctx_vec = get_ctx_vec(sent_ctx_out, sent['tokens'], ent['start'], ent['end'], normalize=False)

                # Skip annotations if the contextual embedding is invalid.
                if np.isnan(span_ctx_vec.sum()) or span_ctx_vec.sum() == 0:
                    continue

                if np.sum(span_ctx_vec) == 0:
                    skipped_anns += 1
                    continue

                # Keep track of the contextual embedding sum for each entity and the number of spans for each entity.
                if ent['cui'] in concept_vecs:
                    concept_vecs[ent['cui']]['vecs_sum'] += span_ctx_vec
                    concept_vecs[ent['cui']]['vecs_num'] += 1
                else:
                    concept_vecs[ent['cui']] = {'vecs_sum': span_ctx_vec, 'vecs_num': 1}

                if ent['st'] in st_ann_vecs:
                    st_ann_vecs[ent['st']]['vecs_sum'] += span_ctx_vec
                    st_ann_vecs[ent['st']]['vecs_num'] += 1
                else:
                    st_ann_vecs[ent['st']] = {'vecs_sum': span_ctx_vec, 'vecs_num': 1}

    logging.info('Skipped %d annotations' % skipped_anns)

    # Store pre-computed entity embeddings.
    logging.info('Writing Concept Vectors ...')
    vecs_path = 'data/processed/mm_st21pv.cuis.%s.vecs' % PYTT_CONFIG['name']
    with open(vecs_path, 'w') as vecs_f:
        # For each entity, compute the entity embedding by averaging the sum of all its embeddings.
        for cui, vecs_info in concept_vecs.items():
            vecs_info['vecs_avg'] = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vecs_info['vecs_avg'].tolist()])
            vecs_f.write('%s %s\n' % (cui, vec_str))
    logging.info('Written %s' % vecs_path)
