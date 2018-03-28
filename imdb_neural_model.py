# Receive input vectors from generator
# Neural network (dense layer) with auto-encoder model for predicting imdb movie scores

import tensorflow as tf
import numpy as np
from keras import backend as K  # Imports core functions compatible with tensors

from keras.models import Model  # Creates model object
from keras.layers import Input, Dense, Activation, core, Dropout
from keras.optimizers import Adam, RMSprop, SGD


# IMDb Training Model
def imdb_regression(size):
    # Input vector length on order of 22,000 elements
    inputs = Input(size)  # Needs tuple input
    dense1 = Dense(16192, name='dense1')(inputs)
    dense2 = Dropout(0.2)(dense1)  # Drop percentage of values to prevent over-fitting
    # Can include as many layers in between (Dense and Dropout)
    dense3 = Dense(8096, name='dense3')(dense2)
    dense4 = Dropout(0.2)(dense3)
    dense5 = Dense(2048, name='dense5')(dense4)
    dense6 = Dropout(0.2, name='dense6')(dense5)
    # Activation - element-wise operation on each output of layer
    # Linear: y=x (specified to be explicit)
    densef = Dense(1, activation='linear', name='densef')(dense6)

    model = Model(input=inputs, output=densef)
    # Optimizer uses Stochastic Gradient Descent; value arbitrary but small
    opt = SGD(lr=0.00001)
    model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['accuracy'])
    print('Successfully generated model')
    return model


# Non-linear learning equivalent of PCA
def auto_encoder(size):
    inputs = Input(size)  # Needs tuple input
    # Reduce dimensionality
    encoder = Dense(10000, activation='relu', name='encoder')(inputs)
    # Bring it back in dimensionality
    autoencoder = Dense(size[0], activation='sigmoid', name='autoencoder')(encoder)

    # Model ties inputs to decoder and attaches weights
    model_autoencoder = Model(input=inputs, output=autoencoder)
    # Instantiate optimizer object
    opt = Adam(lr=0.00001)
    # Model ties inputs to encoder
    model_encoder = Model(input=inputs, output=encoder)
    model_autoencoder.compile(optimizer=opt, loss='mean_absolute_error', metrics=['accuracy'])
    print('Auto-encoder model-generation complete')
    return model_encoder, model_autoencoder
