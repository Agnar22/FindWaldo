from keras.models import Sequential
from tensorflow.keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


def get_conv(input_shape=(64, 64, 3), filename=None):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', input_shape=input_shape, padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (8, 8), activation="relu", name="dense1"))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (14, 14), name="dense2", activation="sigmoid"))
    for layer in model.layers:
        print(layer.input_shape, layer.output_shape)
    if filename:
        model.load_weights(filename)

    model.add(Flatten())
    model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
    return model
