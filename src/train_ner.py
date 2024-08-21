# Load built-in libraries
import time

import numpy as np
# Load external libraries
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load project source code
from configs.load_configs import *
from utils.data_utils import MedMentionsDataset
from utils.train_utils import train, valid


def preprocess_medmentions_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the MedMentions DataFrame.

    Args:
        df (pd.DataFrame): The MedMentions DataFrame.

    Returns:
        pd.DataFrame: The preprocessed MedMentions DataFrame.
    """
    df['sentence'] = df[['Sentence ID', 'Token', 'Tag']].groupby(['Sentence ID'])['Token'].transform(
        lambda x: ' '.join(x))
    df['word_labels'] = df[['Sentence ID', 'Token', 'Tag']].groupby(['Sentence ID'])['Tag'].transform(
        lambda x: ','.join(x))
    return df


if __name__ == "__main__":
    # Print introduction message
    intro = "This script is used to train an NER model on the MedMentions dataset."
    print(intro)

    # Load relevant configurations
    config = get_ner_training_config()
    ner_model_name = config.ner_model_name

    # Check if CUDA is available
    gpu_flag = torch.cuda.is_available()
    device = 'cuda' if gpu_flag else 'cpu'
    print(f"CUDA availability: {gpu_flag}")

    # Load the MedMentions dataset.
    # Set keep_default_na=False to prevent "NaN", "null" etc. from being intepreted as NaN.
    train_path = load_processed_medmentions_ner_path("trng", st21pv_flag=True)
    dev_path = load_processed_medmentions_ner_path("dev", st21pv_flag=True)
    test_path = load_processed_medmentions_ner_path("test", st21pv_flag=True)
    train_dataset = pd.read_csv(train_path, encoding='unicode_escape', keep_default_na=False)
    dev_dataset = pd.read_csv(dev_path, encoding='unicode_escape', keep_default_na=False)
    test_dataset = pd.read_csv(test_path, encoding='unicode_escape', keep_default_na=False)
    train_dataset = preprocess_medmentions_df(train_dataset)
    dev_dataset = preprocess_medmentions_df(dev_dataset)
    test_dataset = preprocess_medmentions_df(test_dataset)
    # medmentions_path = load_processed_medmentions_ner_path(st21pv_flag=True)
    # medmentions_df = pd.read_csv(medmentions_path, encoding='unicode_escape', keep_default_na=False)
    # medmentions_df = preprocess_medmentions_df(medmentions_df)

    # label2id = {k: v for v, k in enumerate(medmentions_df.Tag.unique())}
    # id2label = {v: k for v, k in enumerate(medmentions_df.Tag.unique())}
    label2id = {k: v for v, k in enumerate(train_dataset.Tag.unique())}
    id2label = {v: k for v, k in enumerate(train_dataset.Tag.unique())}

    # medmentions_df = medmentions_df[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
    train_dataset = train_dataset[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
    dev_dataset = dev_dataset[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
    test_dataset = test_dataset[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)

    # train_size = 0.8
    # train_dataset = medmentions_df.sample(frac=train_size, random_state=200)
    # dev_dataset = medmentions_df.drop(train_dataset.index).reset_index(drop=True)
    # train_dataset = train_dataset.reset_index(drop=True)

    # print("FULL Dataset: {}".format(medmentions_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("DEV Dataset: {}".format(dev_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    model = AutoModelForTokenClassification.from_pretrained(ner_model_name,
                                                            num_labels=len(id2label),
                                                            id2label=id2label,
                                                            label2id=label2id
                                                            )
    model.dropout = nn.Dropout(p=config.dropout_prob, inplace=False)
    model.to(device)

    training_set = MedMentionsDataset(train_dataset, tokenizer, config.max_length, label2id)
    dev_set = MedMentionsDataset(dev_dataset, tokenizer, config.max_length, label2id)
    test_set = MedMentionsDataset(test_dataset, tokenizer, config.max_length, label2id)

    train_params = {'batch_size': config.batch_size,
                    'shuffle': config.train_shuffle,
                    'num_workers': config.num_workers
                    }

    dev_params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': config.num_workers
                  }

    test_params = {'batch_size': config.batch_size,
                   'shuffle': False,
                   'num_workers': config.num_workers
                   }

    training_loader = DataLoader(training_set, **train_params)
    dev_loader = DataLoader(dev_set, **dev_params)
    test_loader = DataLoader(test_set, **test_params)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    print("Training starts...")
    start = time.time()
    best_exact_f1 = 0
    best_epoch = 0
    best_loss = np.inf
    for epoch in range(config.num_epochs):
        print(f"Training epoch: {epoch + 1}")
        model, optimizer = train(model, training_loader, optimizer, device, config.max_grad_norm, id2label)
        labels, predictions, ner_labels, ner_preds, entity_level_performance = valid(model, dev_loader, device, id2label)
        if entity_level_performance["exact"]["f1"] > best_exact_f1:
            torch.save(model.state_dict(), config.model_path)
            best_exact_f1 = entity_level_performance["exact"]["f1"]
            best_epoch = epoch + 1
    print("-" * 100)
    model.load_state_dict(torch.load(config.model_path))
    print("Training finished")
    print(f"Best epoch: {best_epoch}; Best f1 score: {best_exact_f1}")
    print("Final verification of results on validation set")
    _ = valid(model, dev_loader, device, id2label)
    print("Evaluation on test set...")
    _ = valid(model, test_loader, device, id2label)
    end = time.time()
    print(f"Training took {(end - start) / 60:.1f} minutes.")
    print("Saving model and tokenizer...")
    torch.save(model.state_dict(), config.model_path)
    tokenizer.save_pretrained(config.tokenizer_path)
    print("Model saved.")
