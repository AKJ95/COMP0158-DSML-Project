import os
import yaml


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
CONFIG_PATH = os.path.join(SRC_ROOT, 'config.yaml')


class NERTrainingConfiguration:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.ner_model_name = config["ner"]["model_name"]
            self.max_length = config["ner"]["max_len"]
            self.batch_size = config["ner"]["batch_size"]
            self.num_epochs = config["ner"]["num_epochs"]
            self.learning_rate = config["ner"]["learning_rate"]
            self.max_grad_norm = config["ner"]["max_grad_norm"]


def load_raw_medmentions_root(st21pv_flag=False) -> str:
    """
    Load the root directory for raw MedMentions data.

    Args:
        st21pv_flag (bool, optional): If True, returns the path to the st21pv version of the data.
                                      If False, returns the path to the full version of the data.
                                      Defaults to False.

    Returns:
        str: The path to the raw MedMentions data.
    """
    version = "medmentions_st21pv_root" if st21pv_flag else "medmentions_full_root"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        mm_root = config["data"]["raw_data"][version]
        return os.path.join(PROJECT_ROOT, mm_root, "corpus_pubtator.txt")


def load_processed_medmentions_ner_path(st21pv_flag=False) -> str:
    """
    Load the root directory for processed MedMentions data.

    Args:
        st21pv_flag (bool, optional): If True, returns the path to the st21pv version of the data.
                                      If False, returns the path to the full version of the data.
                                      Defaults to False.

    Returns:
        str: The path to the processed MedMentions data.
    """
    version = "medmentions_st21pv" if st21pv_flag else "medmentions_full"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        mm_path = config["data"]["processed_data"][version]
        return os.path.join(PROJECT_ROOT, mm_path)


def load_ner_model_name() -> str:
    """
    Load the model name for the NER model.

    Returns:
        str: The model name.
    """
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        return config["ner"]["model_name"]


def get_ner_training_config() -> NERTrainingConfiguration:
    return NERTrainingConfiguration(CONFIG_PATH)
