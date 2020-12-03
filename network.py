import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

SAVE_FILE = "model.h5"

class NNetwork():
    def __init__(self):
        self.model: tf.keras.Model = None

    def load_model(self):
        self.model = keras.models.load_model(SAVE_FILE)

    def train(self):
        dat = np.load("dataset.npz")
        board_states = dat['arr_0']
        evaluations = dat['arr_1']
        
        self.model = keras.models.Sequential()

        self.model.add(layers.Conv2D(8, (3, 3), activation=tf.nn.relu, input_shape=(6, 8, 8)))
        self.model.add(layers.Conv2D(16, (3, 3), activation=tf.nn.relu, data_format='channels_first'))
        self.model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu))
        self.model.add(layers.Conv2D(64, (2, 2), activation=tf.nn.relu))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(500, activation=keras.activations.relu))
        self.model.add(layers.Dense(64, activation=keras.activations.relu))
        self.model.add(layers.Dense(1, activation=keras.activations.linear))

        self.model.compile(
            loss=tf.keras.losses.mean_squared_error,
            optimizer="adam",
            metrics=['accuracy']
        )

        self.model.fit(board_states, evaluations, epochs=10)
        self.model.save(SAVE_FILE)

if __name__ == '__main__':
    NNetwork().train()
