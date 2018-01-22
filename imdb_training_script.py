# Call scripts for input generation and neural network training to produce results

from imdb_input_generator import *
from imdb_neural_model import *

from keras.models import load_model

import pickle as pkl
import h5py
import os

# Link input generator script to neural model script
# Output accuracy of the 3 different models

# Specify directory containing data files:
parent_path = '/Users/cheng-haotai/Documents/Projects_Data/IMDb_Predictor/'

# Load training data files that have been saved:
train1_name = 'train1_index.pickle'
train2_name = 'train2_index.pickle'
train3_name = 'train3_index.pickle'
train1_path = parent_path+train1_name
train2_path = parent_path+train2_name
train3_path = parent_path+train3_name

# Load input data file that has been saved:
input_vector_name = 'input_vectors.h5'
input_vector_path = parent_path+input_vector_name

# Load output data file that has been saved:
output_vector_name = 'output_data.pickle'
output_vector_path = parent_path+output_vector_name

# Get size of network input:
with h5py.File(input_vector_path, 'r') as input_file:
    input_data = input_file['input_dataset'][:]
model_size = (1, len(input_data[0]))
print('Tuple to input into auto encoder function:', model_size)
# Create a model object of the auto-encoder:
encoder, autoencoder = auto_encoder(model_size)

# Load models:
# Weight file details:
weight_folder = parent_path + 'training_weights/'
weight_name_template = 'autoen_weights'
weight_name_ext = '.h5'

# Check if weight folder exists and create if it doesn't:
# Typically all training weights for a model will live in one folder
if not os.path.exists(weight_folder):
    os.mkdir(weight_folder)

# If the weight folder exists, check how many epochs have been processed:
counter = 0
if os.path.exists(weight_folder):
    # Finding the number assigned to most recent epoch
    for epoch in os.listdir(weight_folder):
        e = epoch.split(weight_name_template)
        e = e[1].split(weight_name_ext)
        e = int(e[0])
    if e > counter:
        counter = e
else:
    # Start at epoch 0 if no epochs have been processed
    counter = 0

# Define name of weight file:
weight_file_name = weight_folder + weight_name_template + str(counter) + weight_name_ext
if os.path.exists(weight_file_name):
    # by_name allows for old weight files to be used with new models with new structures
    autoencoder.load_weights(weight_file_name, by_name=True)

# Specify maximum number of epochs to process:
epoch_num = 10
for num in range(counter, epoch_num):
    # Train and generate weights:
    # nb_epoch specifies how many epochs run before saving (1 in this case)
    autoencoder.fit_generator(autoencoder_generator(train1_path, input_vector_path),
                              samples_per_epoch=model_size, nb_epoch=1)
    fresh_weight_name = weight_folder + weight_name_template + str(num) + weight_name_ext
    autoencoder.save_weights(fresh_weight_name)

# Assuming that all epochs have been processed and no need to check last epoch:
# Each weight file is the next progression (weight update) of the previous weight file
# Loading weights trained from autoencoder
# Encoder can "see" weights because by_name = True meaning that trained by name occurred
encoder.load_weights(weight_folder + weight_name_template + str(epoch_num-1) + weight_name_ext)

# Use auto_encoder to encode data into small dimension:
transformed_data = []
for row in input_data:
    placeholder_input = np.zeros((1, model_size))
    placeholder_input[0] = row
    # Gives result of encoder layer in auto_encoder function
    en_predict = encoder.predict(placeholder_input)
    transformed_data.append(en_predict)

# Note: 'epoch' is running through one training data iteration

# Instantiate model
# Load weights
# Fit generator to model
# Save weights
