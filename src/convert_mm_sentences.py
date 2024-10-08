# Code from [1] is consulted and adopted.

import json
import logging
from time import time

import spacy

from mm_reader import read_full_med_mentions
from mm_reader import get_sent_boundaries
from mm_reader import get_sent_ents

sci_nlp = spacy.load('en_core_sci_md')


# Set up logging configuration.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


# Read in raw MedMentions data.
mm_contents = read_full_med_mentions('data/raw/MedMentions-master/st21pv/data/')
mm_splits = {'train': mm_contents[0], 'dev': mm_contents[1], 'test': mm_contents[2]}


logging.info('Processing Instances ...')

# Iterate through each of the three splits of MedMentions.
for split_label in ['dev', 'test', 'train']:
    split_data = {'split': split_label,
                  'timestamp': int(time()),
                  'n_unlocated_mentions': 0,
                  'n_located_mentions': 0,
                  'docs': []}
    instances = mm_splits[split_label]

    # Iterate through each document in the split.
    for doc_idx, ex in enumerate(instances):

        if doc_idx % 100 == 0:
            logging.info('[%s] Converted %d/%d instances.' % (split_label, doc_idx, len(instances)))

        # Record key info for each doc
        doc = {'idx': doc_idx,
               'title': ex.title,
               'abstract': ex.abstract,
               'text': ex.text,
               'pubmed_id': ex.pubmed_id,
               'sentences': []}

        # Get sentence positions to delimit annotations to sentences
        sent_span_idxs = get_sent_boundaries(sci_nlp, ex.text, ex.title)

        # Iterate through each sentence in the document.
        for sent_start, sent_end in sent_span_idxs:
            sent = {}

            # Record tokens of the sentence and the start and end positions of the sentence.
            sent_text = ex.text[sent_start:sent_end + 1]
            sent_tokens = [tok.text.strip() for tok in sci_nlp(sent_text)]
            sent_tokens = [tok for tok in sent_tokens if tok != '']

            sent['text'] = sent_text
            sent['start'] = sent_start
            sent['end'] = sent_end
            sent['tokens'] = sent_tokens

            # Get gold ents
            gold_ents, n_sent_skipped_mentions = get_sent_ents(sci_nlp, sent_tokens, sent_start, sent_end, ex.entities)

            sent['n_unlocated_mentions'] = n_sent_skipped_mentions
            split_data['n_unlocated_mentions'] += n_sent_skipped_mentions

            sent['spans'] = []
            for mm_entity in gold_ents:
                ent = {'cui': mm_entity.cui,
                       'st': mm_entity.st,
                       'tokens': mm_entity.tokens,
                       'start': mm_entity.start,
                       'end': mm_entity.end}
                sent['spans'].append(ent)

            split_data['n_located_mentions'] += len(sent['spans'])
            doc['sentences'].append(sent)

        split_data['docs'].append(doc)
        print(f"Located mentions: {split_data['n_located_mentions']}; Unlocated mentions: {split_data['n_unlocated_mentions']} sentences.")

    # Write the processed MedMentions dataset into a JSON file.
    logging.info('[%s] Writing converted MedMentions ...' % split_label)
    with open('data/processed/mm_converted.%s.json' % split_label, 'w') as json_f:
        json.dump(split_data, json_f, sort_keys=True, indent=4)
