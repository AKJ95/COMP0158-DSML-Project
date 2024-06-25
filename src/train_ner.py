# Load external libraries
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load project source code
from configs.load_configs import *
from utils.data_utils import MedMentionsDataset
from utils.train_utils import *


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
    medmentions_path = load_processed_medmentions_ner_path(st21pv_flag=True)
    medmentions_df = pd.read_csv(medmentions_path, encoding='unicode_escape', keep_default_na=False)
    medmentions_df = preprocess_medmentions_df(medmentions_df)

    label2id = {k: v for v, k in enumerate(medmentions_df.Tag.unique())}
    id2label = {v: k for v, k in enumerate(medmentions_df.Tag.unique())}

    train_size = 0.8
    train_dataset = medmentions_df.sample(frac=train_size, random_state=200)
    test_dataset = medmentions_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(medmentions_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    model = AutoModelForTokenClassification.from_pretrained(ner_model_name,
                                                            num_labels=len(id2label),
                                                            id2label=id2label,
                                                            label2id=label2id
                                                            )
    model.to(device)

    training_set = MedMentionsDataset(train_dataset, tokenizer, config.max_length, label2id)
    testing_set = MedMentionsDataset(test_dataset, tokenizer, config.max_length, label2id)

    train_params = {'batch_size': config.batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': config.batch_size,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        print(f"Training epoch: {epoch + 1}")
        model, optimizer = train(model, training_loader, optimizer, device, config.max_grad_norm)
        labels, predictions, ner_labels, ner_preds = valid(model, testing_loader, device, id2label)
