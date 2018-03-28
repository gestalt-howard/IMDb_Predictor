# Format for feeding into neural network

import numpy as np
import pickle as pkl
import h5py


# Generator for creating network inputs for IMDb
def input_generator(train, output_f, input_f, tdata_name):
    # Open train set:
    with open(train, 'rb') as train_set:
        train_data = pkl.load(train_set)
    # Open input set:
    with h5py.File(input_f, 'r') as input_set:
        input_data = input_set[tdata_name][:]
    # Open output set:
    with open(output_f, 'rb') as output_set:
        output_data = pkl.load(output_set)
    # (input, output)
    input_len = len(input_data[0])
    while 1:
        # While 1 to ensure that generator calls won't ever break
        for t_index in train_data:
            # np.zeros syntax (# elem in list, # sub-elem per elem:
            placeholder_input = np.zeros((1, input_len))
            placeholder_output = np.zeros((1, 1))
            placeholder_input[0] = input_data[t_index]
            placeholder_output[0] = output_data[t_index]
            # Yield keeps track of for loop count and returns only 1 instance per function call
            yield((placeholder_input, placeholder_output))


# Generator for creating auto-encoder inputs
def autoencoder_generator(train, input_f):
    # Open train set:
    with open(train, 'rb') as train_set:
        train_data = pkl.load(train_set)
    # Open input set:
    with h5py.File(input_f, 'r') as input_set:
        input_data = input_set['input_dataset'][:]
    input_len = len(input_data[0])
    while 1:
        for t_index in train_data:
            placeholder = np.zeros((1, input_len))
            placeholder[0] = input_data[t_index]
            yield((placeholder, placeholder))
