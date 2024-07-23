from keras.models import load_model


if __name__ == '__main__':
    model_path = 'models/Classifiers/softmax.cui.h5'
    model = load_model(model_path)
