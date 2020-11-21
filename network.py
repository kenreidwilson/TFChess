import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

SAVE_FILE = "model.h5"

class NNetwork():
    def __init__(self):
        self.model: tf.keras.Model = None

    def load_model(self):
        self.model = keras.models.load_model(SAVE_FILE)

    def train(self):
        activation = "relu"

        self.model = keras.models.Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape = (5, 8, 8)))
        self.model.add(Activation(activation))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(1))
        self.model.add(Activation("tanh"))

        self.model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=['accuracy']
        )

        dat = np.load("processed/dataset.npz")
        board_states = dat['arr_0']
        outcome = dat['arr_1']

        self.model.fit(board_states, outcome, epochs=1)
        self.model.save(SAVE_FILE)

if __name__ == '__main__':
    NNetwork().train()