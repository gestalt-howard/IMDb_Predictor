# Receive input vectors from generator
# Neural network (dense layer) model for predicting imdb movie scores

import tensorflow as tf
import numpy as np
from keras import backend as K  # Imports core functions compatible with tensors

from keras.models import Model  # Creates model object
from keras.layers import Input, Dense, Activation, core, Dropout
from keras.optimizers import Adam, RMSprop, SGD


def imdb_regression((size)):
    inputs = Input((size))
    dense1 = Dense(32768, name='dense1')(inputs)
    dense2 = Dense(1024, name='dense2')(dense1)
    dense2 = Dropout(0.2)(dense2)  # Drop percentage of values to prevent overfitting
    # Can include as many layers in between (Dense and Dropout)
    densef = Dense(1, activation='linear', name='densef')(dense2)

    model = Model(input=inputs, output=densef)
    opt = SGD(lr=0.00001)
    model.compile(optimizer=opt, loss='mean absolute error', metrics=['accuracy'])
    print('Successfully generated model')
    return model


