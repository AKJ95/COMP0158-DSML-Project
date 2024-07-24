from keras.models import Sequential
from keras.layers import Dense


if __name__ == '__main__':
    model = Sequential([
        Dense(18426, activation='softmax', input_shape=(768,)),
    ])
    model.load_weights("models/Classifiers/softmax.cui.h5")

