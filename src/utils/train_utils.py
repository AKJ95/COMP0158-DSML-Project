# Import built-in libraries
from typing import Tuple

# Import external libraries
from nervaluate import Evaluator
import torch
from torchmetrics import Accuracy


def get_batch_data(batch, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = batch['ids'].to(device, dtype=torch.long)
    mask = batch['mask'].to(device, dtype=torch.long)
    targets = batch['targets'].to(device, dtype=torch.long)
    return ids, mask, targets


def forward_pass(model, ids, mask, targets) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
    loss, logits = outputs.loss, outputs.logits
    return loss, logits


def compute_accuracy(model, targets, logits, mask) -> Tuple[torch.Tensor, torch.Tensor]:
    flattened_targets = targets.view(-1)
    active_logits = logits.view(-1, model.num_labels)
    flattened_predictions = torch.argmax(active_logits, dim=1)
    active_accuracy = mask.view(-1) == 1
    targets = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)
    return targets, predictions


def compute_entity_level_performance(labels: list[list[str]], predictions: list[list[str]]) -> dict:
    evaluator = Evaluator(labels, predictions, tags=["Entity"], loader="list")
    performance_dict, _, _, _ = evaluator.evaluate()
    return performance_dict


def transpose_and_flatten_2d_list(lst):
    transposed = [list(i) for i in zip(*lst)]
    flattened = [item for sublist in transposed for item in sublist if item != '[PAD]']
    return flattened


# Defining the training function on the 80% of the dataset for tuning the bert model
def train(model, training_loader, optimizer, device, max_grad_norm, id2label) -> Tuple[torch.nn.Module,
                                                                                       torch.optim.Optimizer]:
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    nervaluate_preds, nervaluate_labels = [], []
    # put model in training mode
    model.train()

    for idx, batch in enumerate(training_loader):
        ids, mask, targets = get_batch_data(batch, device)
        loss, tr_logits = forward_pass(model, ids, mask, targets)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if idx % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss per 100 training steps at step {idx}: {loss_step}")

        targets, predictions = compute_accuracy(model, targets, tr_logits, mask)

        tr_preds.extend(predictions)
        tr_labels.extend(targets)

        # print(ids.shape)
        # print(ids[0])
        # print(mask[0])
        # print(targets.shape)
        # print(predictions.shape)
        # print(len(batch['tokens']))
        tokens = transpose_and_flatten_2d_list(batch['tokens'])
        word_level_predictions = []
        word_level_targets = []
        wp_preds = list(zip(tokens, predictions))
        for index in range(len(wp_preds)):
            if (wp_preds[index][0].startswith("##")) or (wp_preds[index][0] in ['[CLS]', '[SEP]', '[PAD]']):
                # skip prediction
                continue
            else:
                word_level_predictions.append(wp_preds[index][1])
                word_level_targets.append(targets[index])
        # print(len(word_level_predictions))
        # print(len(word_level_targets))
        # print(type(word_level_targets[0]))
        nervaluate_labels.append([id2label[tag_id.item()] for tag_id in word_level_targets])
        nervaluate_preds.append([id2label[tag_id.item()] for tag_id in word_level_predictions])

        accuracy_score = Accuracy(task="multiclass", num_classes=model.num_labels)
        tmp_tr_accuracy = accuracy_score(targets.cpu(), predictions.cpu())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")
    entity_level_performance = compute_entity_level_performance(nervaluate_labels, nervaluate_preds)
    print("Entity level performance: ", entity_level_performance)
    return model, optimizer


def valid(model, testing_loader, device, id2label) -> Tuple[list, list, list, list, dict]:
    """
    Function to evaluate the model on the dataset
    """
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    nervaluate_preds, nervaluate_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids, mask, targets = get_batch_data(batch, device)
            loss, eval_logits = forward_pass(model, ids, mask, targets)
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                print(f"Validation loss per 100 evaluation steps at step {idx}: {loss_step}")

            # compute evaluation accuracy
            targets, predictions = compute_accuracy(model, targets, eval_logits, mask)

            eval_labels.extend(targets)
            eval_preds.extend(predictions)

            tokens = transpose_and_flatten_2d_list(batch['tokens'])
            word_level_predictions = []
            word_level_targets = []
            wp_preds = list(zip(tokens, predictions))
            for index in range(len(wp_preds)):
                if (wp_preds[index][0].startswith("##")) or (wp_preds[index][0] in ['[CLS]', '[SEP]', '[PAD]']):
                    # skip prediction
                    continue
                else:
                    word_level_predictions.append(wp_preds[index][1])
                    word_level_targets.append(targets[index])
            # print(word_level_predictions[:50])
            # print(word_level_targets[:50])
            nervaluate_labels.append([id2label[tag_id.item()] for tag_id in word_level_targets])
            nervaluate_preds.append([id2label[tag_id.item()] for tag_id in word_level_predictions])

            accuracy_score = Accuracy(task="multiclass", num_classes=model.num_labels)
            tmp_eval_accuracy = accuracy_score(targets.cpu(), predictions.cpu())
            eval_accuracy += tmp_eval_accuracy

    labels = [id2label[tag_id.item()] for tag_id in eval_labels]
    predictions = [id2label[tag_id.item()] for tag_id in eval_preds]
    print(len(nervaluate_labels))
    print(len(nervaluate_labels[0]))

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    entity_level_performance = compute_entity_level_performance(nervaluate_labels, nervaluate_preds)
    print("Entity level performance: ", entity_level_performance)

    return labels, predictions, nervaluate_labels, nervaluate_preds, entity_level_performance
