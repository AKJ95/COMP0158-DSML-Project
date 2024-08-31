import json


def read_mm_converted(mm_set_path):
    with open(mm_set_path, 'r') as json_f:
        mm_set = json.load(json_f)

    return list(mm_set['docs'])

# Open the file
with open('results/str_1nn_rerank.txt', 'r') as f:
    # Parse the JSON string
    data = json.load(f)

# Now 'data' is a Python dictionary that you can work with
gold_labels = data['gold_labels']
pred_labels = data['pred_labels']


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

num_spans_train = 0
gold_labels_train = []
mm_docs = read_mm_converted('data/processed/mm_converted.train.json')
for doc in mm_docs:
    for sentence in doc['sentences']:
        num_spans_train += len(sentence['spans'])
        for span in sentence['spans']:
            gold_labels_train.append(span['cui'])
print("Number of spans in train set:", num_spans_train)
assert len(gold_labels) == num_spans_train
print(gold_labels_train[:10])
