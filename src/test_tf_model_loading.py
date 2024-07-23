import torch


if __name__ == '__main__':
    model_path = 'models/Classifiers/softmax.cui.h5'
    model = torch.load(model_path)
