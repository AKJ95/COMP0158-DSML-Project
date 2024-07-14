import json
import logging

# from matcher_exactmatch import WhitespaceTokenizer  # ???

from NERComponent import NERComponent
from medlinker import MedLinker
# st21pv
from umls import umls_kb_st21pv as umls_kb

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logging.info('Loading MedLinker ...')


cx_ner_path = 'models/ContextualNER/mm_st21pv_SCIBERT_uncased/'
em_ner_path = 'models/ExactMatchNER/umls.2017AA.active.st21pv.nerfed_nlp_and_matcher.max3.p'
# ngram_db_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.3gram.5toks.db'
# ngram_map_path = 'models/SimString/umls.2017AA.active.st21pv.aliases.5toks.map'
ngram_db_path = 'data/processed/umls.2024AA.active.st21pv.aliases.3gram.5toks.db'
ngram_map_path = 'data/processed/umls.2024AA.active.st21pv.aliases.5toks.map'
st_vsm_path = 'models/VSMs/mm_st21pv.sts_anns.scibert_scivocab_uncased.vecs'
# cui_vsm_path = 'models/VSMs/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
cui_vsm_path = 'data/processed/mm_st21pv.cuis.scibert_scivocab_uncased.vecs'
cui_idx_path = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.index'
cui_lbs_path = 'models/VSMs/umls.2017AA.active.st21pv.scibert_scivocab_uncased.cuis.labels'

# full
# from umls import umls_kb_full as umls_kb
# cx_ner_path = 'models/ContextualNER/mm_full_SCIBERT_uncased/'
# ngram_db_path = 'models/SimString/umls.2017AA.active.full.aliases.3gram.5toks.db'
# ngram_map_path = 'models/SimString/umls.2017AA.active.full.aliases.5toks.map'


print('Loading MedNER ...')
# medner = MedNER(cx_ner_path, em_ner_path)
medner = NERComponent()

print('Loading MedLinker ...')
medlinker = MedLinker(medner, umls_kb)
# medlinker.load_st_VSM(st_vsm_path)
medlinker.load_string_matcher(ngram_db_path, ngram_map_path)
# medlinker.load_cui_FaissVSM(cui_idx_path, cui_lbs_path)
# medlinker.load_cui_VSM(cui_vsm_path)


# input('...')

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
    # mm_docs = read_mm_converted('data/MedMentions/full/custom/mm_converted.dev.json')
    # mm_docs = read_mm_converted('data/MedMentions/st21pv/custom/mm_converted.dev.json')
    mm_docs = read_mm_converted('data/processed/mm_converted.test.json')

    logging.info('Processing Instances ...')
    for doc_idx, doc in enumerate(mm_docs):
        perf_stats['n_docs'] += 1

        logging.info('At doc #%d' % doc_idx)

        gold_ents = set()
        for gold_sent in doc['sentences']:
            for gold_span in gold_sent['spans']:
                gold_ents.add(gold_span['cui'].lstrip('UMLS:'))

        pred_ents = set()
        for gold_sent in doc['sentences']:
            sent_preds = medlinker.predict(' '.join(gold_sent['tokens']))
            for pred_span in sent_preds['spans']:
                if pred_span['cui'] is not None:
                    pred_ents.add(pred_span['cui'][0])

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
