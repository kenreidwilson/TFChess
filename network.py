import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input

SAVE_FILE = "model.h5"

class NNetwork():
    def __init__(self):
        self.model: tf.keras.Model = None

    def load_model(self):
        self.model = keras.models.load_model(SAVE_FILE)

    def train(self):
        dat = np.load("processed/dataset.npz")
        board_states = dat['arr_0']
        outcomes = dat['arr_1']

        self.model = keras.models.Sequential()

        self.model.add(Input(64))
        self.model.add(Dense(1048, activation=tf.nn.relu))
        self.model.add(Dense(500, activation=tf.nn.relu))
        self.model.add(Dense(50, activation=tf.nn.relu))
        self.model.add(Dense(1, activation=keras.activations.linear))

        self.model.compile(
            loss=tf.keras.losses.mean_squared_error,
            optimizer="adam",
            metrics=['accuracy']
        )

        self.model.fit(board_states, outcomes, epochs=10)
        self.model.save(SAVE_FILE)

if __name__ == '__main__':
    NNetwork().train()
