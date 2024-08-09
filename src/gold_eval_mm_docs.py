import json
import logging

import spacy

from NERComponent import NERComponent
from medlinker import MedLinker
from umls import umls_kb_full as umls_kb

sci_nlp = spacy.load('en_core_sci_md', disable=['tagger', 'parser', 'ner'])

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logging.info('Loading MedLinker ...')


cx_ner_path = 'models/ContextualNER/mm_st21pv_SCIBERT_uncased/'
ngram_db_path = 'data/processed/umls.2024AA.active.st21pv.aliases.3gram.5toks.db'
ngram_map_path = 'data/processed/umls.2024AA.active.st21pv.aliases.5toks.map'
cui_vsm_path = 'data/processed/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
cui_def_vsm_path = 'data/processed/umls.2024AA.active.st21pv.scibert_scivocab_uncased.cuis.vecs'
cui_idx_path = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.index'
cui_lbs_path = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.labels'


print('Loading MedNER ...')
medner = NERComponent()

print('Loading MedLinker ...')
medlinker = MedLinker(medner, umls_kb)
medlinker.load_string_matcher(ngram_db_path, ngram_map_path)
# medlinker.load_cui_softmax_pt()
medlinker.load_cui_VSM(cui_vsm_path)


def read_mm_converted(mm_set_path):
    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])


def calc_p(metrics):
    try:
        return metrics['tp'] / (metrics['tp'] + metrics['fp'])
    except ZeroDivisionError:
        return 0


def calc_r(metrics):
    try:
        return metrics['tp'] / (metrics['tp'] + metrics['fn'])
    except ZeroDivisionError:
        return 0


def calc_f1(metrics):
    try:
        p = calc_p(metrics)
        r = calc_r(metrics)
        return 2 * ((p * r) / (p + r))
    except ZeroDivisionError:
        return 0


def calc_acc(metrics):
    try:
        # return metrics['tp'] / sum(metrics.values())
        return metrics['tp'] / (metrics['tp'] + metrics['fp'] + metrics['fn'])
    except ZeroDivisionError:
        return 0


def calc_counts(metrics):
    metrics['n'] = sum(metrics.values())
    return metrics


def stringify_metrics(metrics):
    metrics_counts = calc_counts(metrics)
    return ' '.join(['%s:%d' % (l.upper(), c) for l, c in metrics_counts.items()])


if __name__ == '__main__':

    perf_stats = {'n_gold_spans': 0, 'n_pred_spans': 0, 'n_sents': 0, 'n_docs': 0}
    perf_cui = {'tp': 0, 'fp': 0, 'fn': 0}

    logging.info('Loading MedMentions ...')
    mm_docs = read_mm_converted('data/processed/mm_converted.test.json')

    logging.info('Processing Instances ...')
    span_count = 0
    in_top_n_count = 0
    skip_count = 0
    for doc_idx, doc in enumerate(mm_docs):
        perf_stats['n_docs'] += 1

        logging.info('At doc #%d' % doc_idx)

        gold_ents = set()
        for gold_sent in doc['sentences']:
            for gold_span in gold_sent['spans']:
                gold_ents.add(gold_span['cui'].lstrip('UMLS:'))

        pred_ents = set()
        for gold_sent in doc['sentences']:
            gold_spans = [(span['start'], span['end']) for span in gold_sent['spans']]
            sent_preds = medlinker.predict(' '.join(gold_sent['tokens']),
                                           gold_tokens=gold_sent['tokens'],
                                           gold_spans=gold_spans,
                                           top_n=10)
            for i in range(len(sent_preds['spans'])):
                embedding_tokens = []
                embedding_tokens.extend(gold_sent['tokens'][:sent_preds['spans'][i]['start']])
                embedding_tokens.append('[M_s]')
                embedding_tokens.extend(gold_sent['tokens'][sent_preds['spans'][i]['start']:sent_preds['spans'][i]['end']])
                embedding_tokens.append('[M_e]')
                embedding_tokens.extend(gold_sent['tokens'][sent_preds['spans'][i]['end']:])

                gold_entity_cui = gold_sent['spans'][i]['cui'].lstrip('UMLS:')
                gold_entity_kb = umls_kb.get_entity_by_cui(gold_sent['spans'][i]['cui'].lstrip('UMLS:'))
                gold_entity_name = gold_entity_kb['Name'] if gold_entity_kb else ''

                span_count += 1
                pred_entities = [entry[0] for entry in sent_preds['spans'][i]['cui']]
                gold_entity = gold_sent['spans'][i]['cui'].lstrip('UMLS:')
                if gold_entity in pred_entities:
                    in_top_n_count += 1
            for pred_span in sent_preds['spans']:
                for pred_cui in pred_span['cui']:
                    pred_ents.add(pred_cui[0])

        perf_cui['tp'] += len(gold_ents.intersection(pred_ents))
        perf_cui['fp'] += len([pred_ent for pred_ent in pred_ents if pred_ent not in gold_ents])
        perf_cui['fn'] += len([gold_ent for gold_ent in gold_ents if gold_ent not in pred_ents])

        # in-progress performance metrics
        p = calc_p(perf_cui) * 100
        r = calc_r(perf_cui) * 100
        f = calc_f1(perf_cui) * 100
        a = calc_acc(perf_cui) * 100

        counts = calc_counts(perf_cui)
        counts_str = '\t'.join(['%s:%d' % (l.upper(), c) for l, c in counts.items()])
        print('[CUI]\tP:%.2f\tR:%.2f\tF1:%.2f\tACC:%.2f - %s' % (p, r, f, a, counts_str))
        print(f"Recall per span: {in_top_n_count / span_count * 100:.2f}%")
        print(f"Skipped: {skip_count}/{span_count} ({skip_count / span_count * 100:.2f}%)")
