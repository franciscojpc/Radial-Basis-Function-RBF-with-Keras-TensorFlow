"""
Radial Basis Networks and Custom Keras Layers
From 
https://www.kaggle.com/code/residentmario/radial-basis-networks-and-custom-keras-layers/notebook
"""


from pickletools import optimize
from random import shuffle
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Layer

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import binary_crossentropy

class RBFLayer(Layer):
    def __init__(self, units, gamma, **args) -> None:
        super(RBFLayer, self).__init__(**args)
        self.units = units
        self.gamma = gamma

    def build(self, input_shape):
        self.mu = self.add_weight(
            name='mu',
            shape=(int(input_shape[1]), self.units),
            initializer='uniform',
            trainable=True
        )
        super(RBFLayer, self).build(input_shape)
    
    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1*self.gamma*l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)



X = np.load('DB/k49-train-imgs.npz')['arr_0']/255

y = np.load('DB/k49-train-labels.npz')['arr_0']
y = (y <= 25).astype(int)


if __name__ == '__main__':
    gamma = 0.5
    units = int(28*28/2)
    epochs = 10
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(RBFLayer(units=units, gamma=gamma))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss=binary_crossentropy)

    model.fit(X, y, batch_size=256, epochs=epochs, shuffle=True)