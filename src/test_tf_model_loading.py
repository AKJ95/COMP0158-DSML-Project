import joblib


if __name__ == '__main__':
    model_path = 'models/Classifiers/softmax.cui.h5'
    model = joblib.load(model_path)
