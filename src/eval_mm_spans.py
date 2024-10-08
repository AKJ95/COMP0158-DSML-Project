# Code from [1] is consulted and adopted.

import json
import logging

from umls import umls_kb_st21pv_2017 as umls_kb
from NERComponent import NERComponent
from medlinker import MedLinker

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

gold_labels = []
pred_labels = []


def read_mm_converted(mm_set_path):
    """
    Read the MedMentions dataset from a json file.
    :param mm_set_path: filepath to the MedMentions dataset
    :return: Content of the MedMentions dataset.
    """
    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])


# Helper functions to compute performance metrics.
def calc_metrics(obs):
    #
    def calc_p(obs):
        try:
            return len(obs['tp']) / (len(obs['tp']) + len(obs['fp']))
        except ZeroDivisionError:
            return 0

    def calc_r(obs):
        try:
            return len(obs['tp']) / (len(obs['tp']) + len(obs['fn']))
        except ZeroDivisionError:
            return 0

    def calc_f1(obs):
        try:
            p = calc_p(obs)
            r = calc_r(obs)
            return 2 * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0

    def calc_acc(obs):
        try:
            return len(obs['tp']) / sum([len(v) for v in obs.values()])
        except ZeroDivisionError:
            return 0

    p = calc_p(obs) * 100
    r = calc_r(obs) * 100
    f1 = calc_f1(obs) * 100
    acc = calc_acc(obs) * 100

    return p, r, f1, acc


# Helper function to print performance metrics.
def stringify_obs(obs):
    obs_counts = {k: len(m) for k, m in obs.items()}
    obs_counts['n'] = sum(obs_counts.values())
    return ' '.join(['%s:%d' % (l.upper(), c) for l, c in obs_counts.items()])


# Helper function to update performance metrics.
def update_obs(doc_idx, sent_idx, gold_spans, pred_spans, perf_ner, perf_st, perf_cui):
    # 1st pass - register pred matched and unmatched (TP & FP)
    for pred_span in pred_spans:
        pred_start, pred_end = pred_span['start'], pred_span['end']
        pred_info = (doc_idx, sent_idx, pred_start, pred_end)

        matched_ner, matched_st, matched_cui = False, False, False
        for gold_span in gold_spans:
            gold_start, gold_end = gold_span['start'], gold_span['end']
            gold_info = (doc_idx, sent_idx, gold_start, gold_end)

            # print(pred_start, pred_end)
            # print(gold_start, gold_end)
            if (pred_start == gold_start) and (pred_end == gold_end):
                matched_ner = True

                if pred_span['st'] is not None:
                    if pred_span['st'][0] == gold_span['st']:
                        matched_st = True  # matched st & NER

                if pred_span['cui'] is not None:
                    gold_span['cui'] = gold_span['cui'].lstrip('UMLS:')
                    if pred_span['cui'][0][0] == gold_span['cui']:
                        matched_cui = True  # matched cui & NER

        if matched_ner:
            perf_ner['tp'].add(pred_info)
        else:
            perf_ner['fp'].add(pred_info)

        if matched_st:
            perf_st['tp'].add(pred_info)
        else:
            perf_st['fp'].add(pred_info)

        if matched_cui:
            perf_cui['tp'].add(pred_info)
        else:
            perf_cui['fp'].add(pred_info)

    # 2nd pass - register unmatched preds (FN)
    for gold_span in gold_spans:
        gold_start, gold_end = gold_span['start'], gold_span['end']
        gold_info = (doc_idx, sent_idx, gold_start, gold_end)

        # print(gold_info)
        # print(perf_cui['tp'].union(perf_cui['fp']))
        if gold_info not in perf_ner['tp'].union(perf_ner['fp']):
            perf_ner['fn'].add(gold_info)

        if gold_info not in perf_st['tp'].union(perf_st['fp']):
            perf_st['fn'].add(gold_info)

        if gold_info not in perf_cui['tp'].union(perf_cui['fp']):
            # print(gold_info)
            perf_cui['fn'].add(gold_info)

        store_results = True
        if use_gold_spans and store_results:
            found_pred = False
            gold_labels.append(gold_span['cui'].lstrip('UMLS:'))
            for pred_span in pred_spans:
                pred_start, pred_end = pred_span['start'], pred_span['end']
                pred_info = (doc_idx, sent_idx, pred_start, pred_end)
                if gold_info == pred_info:
                    pred_labels.append(pred_span['cui'][0][0])
                    found_pred = True
                    break
            if not found_pred:
                pred_labels.append(None)

    # print(len(gold_labels), len(pred_labels))


if __name__ == '__main__':

    use_gold_spans = True
    mm_ann = 'cui'

    # File paths related to the predictors and meta learner of MedLinker.
    ngram_db_path = 'data/processed/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
    ngram_map_path = 'data/processed/umls.2017AA.active.st21pv.aliases.5toks.map'
    cui_vsm_path = 'data/processed/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
    cui_val_path = 'models/Validators/mm_st21pv.lr_clf_cui.final.joblib'

    # Mention detection component.
    print('Loading MedNER ...')
    medner = NERComponent()

    # Load MedLinker and configure it according to need, i.e. load the necessary predictors.
    print('Loading MedLinker ...')
    medlinker = MedLinker(medner, umls_kb)
    # medlinker.load_string_matcher(ngram_db_path, ngram_map_path)

    predict_cui, require_cui = False, False
    predict_sty, require_sty = False, False
    if mm_ann == 'cui':
        print("Predicting for CUIs...")
        # medlinker.load_cui_VSM(cui_vsm_path)
        medlinker.load_cui_softmax_pt()
        # medlinker.load_cui_validator(cui_val_path, validator_thresh=0.5)

        predict_cui, require_cui = True, True

    # Obsolete
    elif mm_ann == 'sty':
        # medlinker.load_st_VSM(st_vsm_path)
        # medlinker.load_sty_clf(sty_clf_path)
        # sty_val_path = 'models/Validators/mm_st21pv.lr_clf_sty.dev.joblib'
        # medlinker.load_st_validator(sty_val_path, validator_thresh=0.45)

        predict_sty, require_sty = True, True

    perf_stats = {'n_gold_spans': 0, 'n_pred_spans': 0, 'n_sents': 0, 'n_docs': 0}
    perf_ner = {'tp': set(), 'fp': set(), 'fn': set()}
    perf_cui = {'tp': set(), 'fp': set(), 'fn': set()}
    perf_st = {'tp': set(), 'fp': set(), 'fn': set()}

    # Load MedMentions test split.
    logging.info('Loading MedMentions ...')
    mm_docs = read_mm_converted('data/processed/mm_converted.test.json')

    # Iterate through MedMentions test set.
    logging.info('Processing Instances ...')
    for doc_idx, doc in enumerate(mm_docs):
        perf_stats['n_docs'] += 1

        logging.info('At doc #%d' % doc_idx)

        # Iterate through all sentences of the document.
        for sent_idx, gold_sent in enumerate(doc['sentences']):
            perf_stats['n_sents'] += 1

            # Use gold spans or detect spans using the mention detector component.
            # Then, predict for the spans using MedLinker.
            if use_gold_spans:
                gold_spans = [(s['start'], s['end']) for s in gold_sent['spans']]
                gold_tokens = gold_sent['tokens']

                preds = medlinker.predict(sentence=' '.join(gold_sent['tokens']),
                                          gold_tokens=gold_tokens, gold_spans=gold_spans,
                                          predict_cui=predict_cui, predict_sty=predict_sty,
                                          require_cui=require_cui, require_sty=require_sty)
                # assert len(gold_sent['spans']) == len(preds['spans'])

            else:
                preds = medlinker.predict(sentence=' '.join(gold_sent['tokens']),  # expects ws separated text
                                          predict_cui=predict_cui, predict_sty=predict_sty,
                                          require_cui=require_cui, require_sty=require_sty)

            # Update performance metrics based on predictions.
            pred_spans = preds['spans']
            gold_spans = gold_sent['spans']
            # assert preds['tokens'] == gold_sent['tokens']  # hence, equal boundaries == equal text

            perf_stats['n_gold_spans'] += len(gold_spans)
            perf_stats['n_pred_spans'] += len(pred_spans)

            update_obs(doc_idx, sent_idx, gold_spans, pred_spans, perf_ner, perf_st, perf_cui)

        # Display performance metrics.
        for pred_type, type_obs in [('NER', perf_ner), ('STY', perf_st), ('CUI', perf_cui)]:
            p, r, f1, acc = calc_metrics(type_obs)
            obs_str = stringify_obs(type_obs)
            print('[%s] P:%.2f R:%.2f F1:%.2f ACC:%.2f - %s' % (pred_type, p, r, f1, acc, obs_str))
        print(perf_stats)
        print()

    # Double-check performance metrics by calculating them from the stored results.
    print("Analysing from stored results...")
    tp = 0
    fp = 0
    fn = pred_labels.count(None)
    for i in range(len(pred_labels)):
        if pred_labels[i] is not None:
            if pred_labels[i] == gold_labels[i]:
                tp += 1
            else:
                fp += 1
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * ((p * r) / (p + r))
    print("P:", p)
    print("R:", r)
    print("F1:", f1)

    # Save results if evaluation is conducted using gold spans.
    if gold_labels and pred_labels:
        # Convert the list to a JSON string
        results = {"gold_labels": gold_labels, "pred_labels": pred_labels}
        results_str = json.dumps(results)
        # Write the string to a file
        with open('results/clf.txt', 'w') as file:
            file.write(results_str)
