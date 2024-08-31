import json

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
