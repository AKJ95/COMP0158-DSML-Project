# Code from [1] is consulted and adopted.


import json
import logging
import numpy as np
import itertools

from sklearn.linear_model import LogisticRegression
import joblib
import torch

# from matcher_exactmatch import WhitespaceTokenizer  # ???

from NERComponent import NERComponent
from medlinker import MedLinker
from medlinker import MedLinkerDoc

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logging.info('Loading MedLinker ...')

# st21pv
from umls import umls_kb_st21pv_2017 as umls_kb

ngram_db_path = 'data/processed/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
ngram_map_path = 'data/processed/umls.2017AA.active.st21pv.aliases.5toks.map'
cui_vsm_path = 'models/VSMs/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'

print('Loading MedNER ...')
medner = NERComponent()

print('Loading MedLinker ...')
medlinker = MedLinker(medner, umls_kb)
medlinker.load_string_matcher(ngram_db_path, ngram_map_path)
medlinker.load_cui_VSM(cui_vsm_path)
medlinker.load_cui_softmax_pt()


def read_mm_converted(mm_set_path):
    """
    Read the MedMentions dataset from a json file.
    :param mm_set_path: Filepath to the MedMentions dataset json file.
    :return: Return the contents of the MedMentions dataset in a list.
    """
    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])


if __name__ == '__main__':

    # Load MedMentions training set
    logging.info('Loading MedMentions ...')
    mm_docs = read_mm_converted('data/processed/mm_converted.train.json')
    # mm_docs = read_mm_converted('data/MedMentions/st21pv/custom/mm_converted.dev.json')

    X_sty, X_cui, y_sty, y_cui = [], [], [], []
    n_skipped = 0

    # Iterate through the training set and create training instances
    logging.info('Processing Instances ...')
    for doc_idx, doc in enumerate(mm_docs):

        logging.info('At doc #%d - len(X)=%d, n_skipped=%d' % (doc_idx, len(X_cui), n_skipped))

        for sent_idx, gold_sent in enumerate(doc['sentences']):

            gold_spans = [(s['start'], s['end']) for s in gold_sent['spans']]

            medlinker_doc = MedLinkerDoc(text=' '.join(gold_sent['tokens']),
                                         tokens=gold_sent['tokens'],
                                         spans=gold_spans)

            medlinker_doc.set_contextual_vectors()

            # for span_start, span_end, span_vec in medlinker_doc.get_spans(include_vectors=True, normalize=True):
            for span_start, span_end, span_vec in medlinker_doc.get_spans(include_vectors=True, normalize=False):
                span_str = ' '.join(medlinker_doc.tokens[span_start:span_end])

                # CUI Matching
                matches_cui_str = medlinker.string_matcher.match_cuis(span_str.lower())
                # matches_cui_vsm = medlinker.cui_vsm.most_similar(span_vec)
                span_vec_tensor = torch.unsqueeze(torch.from_numpy(span_vec), 0)
                span_vec_tensor = span_vec_tensor.to(medlinker.device)
                matches_cui_vsm = medlinker.cui_clf.predict(span_vec_tensor, threshold=0.5)

                if len(matches_cui_str) + len(matches_cui_vsm) == 0:
                    n_skipped += 1
                    continue

                cui_matchers_agree = False
                if len(matches_cui_str) > 0 and len(matches_cui_vsm) > 0:
                    if matches_cui_str[0][0] == matches_cui_vsm[0][0]:
                        cui_matchers_agree = True

                scores_cui_str = dict(matches_cui_str)
                scores_cui_vsm = dict(matches_cui_vsm)

                cui_matches = {cui: max(scores_cui_str.get(cui, 0), scores_cui_vsm.get(cui, 0))
                               for cui in scores_cui_str.keys() | scores_cui_vsm.keys()}
                cui_matches = sorted(cui_matches.items(), key=lambda x: x[1], reverse=True)
                cui_top_match = cui_matches[0][0]

                # CUI Features
                x_cui = []
                if len(matches_cui_str) > 0:
                    x_cui.append(matches_cui_str[0][1])
                else:
                    x_cui.append(0)
                if len(matches_cui_vsm) > 0:
                    x_cui.append(matches_cui_vsm[0][1])
                else:
                    x_cui.append(0)
                x_cui.append(cui_matches[0][1])
                x_cui.append((scores_cui_str.get(cui_top_match, 0) + scores_cui_vsm.get(cui_top_match, 0)) / 2)
                # x_cui.append(int(sty_matchers_agree))
                x_cui.append(int(cui_matchers_agree))
                X_cui.append(x_cui)

                #
                cui_pred_correct = False
                for gold_span in gold_sent['spans']:
                    if span_start == gold_span['start'] and span_end == gold_span['end']:
                        if cui_top_match == gold_span['cui'].lstrip('UMLS:'):
                            cui_pred_correct = True
                            break

                y_cui.append(int(cui_pred_correct))

    # Train and save the logistical regression meta learner.
    logging.info('Training CUI Logistic Regression ...')
    cui_lr_clf = LogisticRegression(random_state=42, verbose=True)
    cui_lr_clf.fit(X_cui, y_cui)

    logging.info('Saving CUI Classifier ...')
    joblib.dump(cui_lr_clf, 'models/Validators/mm_st21pv.lr_clf_cui.final.joblib')
