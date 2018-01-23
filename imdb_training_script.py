# Call scripts for input generation and neural network training to produce results

from imdb_input_generator import *
from imdb_neural_model import *

import pickle as pkl
import h5py
import os
import pandas as pd

import keras
from keras import backend as K
if keras.__version__[0] == '1':
    K.set_image_dim_ordering('th')
else:
    K.set_image_data_format('channels_first')

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

# Load test index data files that have been saved:
test_name = 'test_index.pickle'
test_name_path = parent_path+test_name

##########
# Auto-Encoder Portion:
print('Starting auto-encoder portion:')
# Get size of network input:
with h5py.File(input_vector_path, 'r') as input_file:
    input_data = input_file['input_dataset'][:]
# model_size = (1, len(input_data[0]))
model_len = len(input_data[0])
model_size = (model_len, )  # Tuple of size 1
print('Tuple to input into auto encoder function has size:', model_size)
# Create a model object of the auto-encoder:
print('Generating model for encoder and auto-encoder:')
encoder, autoencoder = auto_encoder(model_size)

# Load or create weight files for autoencoder:
# Weight file details:
weight_folder = parent_path + 'autoen_training_weights/'
weight_name_template = 'autoen_weights'
weight_name_ext = '.h5'

# Check if weight folder exists and create if it doesn't:
# Typically all training weights for a model will live in one folder
if not os.path.exists(weight_folder):
    print('Creating folder for auto-encoder weights...')
    os.mkdir(weight_folder)
else:
    print('Folder for auto-encoder weights already exists!')

# Note: 'epoch' is running through one training data iteration
# If the weight folder exists, check how many epochs have been processed:
counter = -1
if os.path.exists(weight_folder):
    # Finding the number assigned to most recent epoch
    for epoch in os.listdir(weight_folder):
        e = epoch.split(weight_name_template)
        e = e[1].split(weight_name_ext)
        e = int(e[0])
        if e > counter:
            counter = e
    print('Most up-to-date epoch is indexed:', counter)
else:
    # Start at epoch 0 if no epochs have been processed
    counter = -1
    print('No epochs for auto-encoder have run yet')

# Define name of weight file:
weight_file_name = weight_folder + weight_name_template + str(counter) + weight_name_ext
if os.path.exists(weight_file_name):
    print('Latest auto-encoder weight file already exists! Loading...')
    # by_name allows for old weight files to be used with new models with new structures
    autoencoder.load_weights(weight_file_name, by_name=True)
else:
    print('No weights have been generated yet. Proceeding to generate weights...')

# Specify maximum number of epochs to process:
epoch_num = 10
for num in range(counter+1, epoch_num):
    print('Auto-encoder weight generating on epoch', num)
    # Train (implies updating error) and generate weights:
    # nb_epoch specifies how many epochs run before saving (1 in this case)
    # samples_per_epoch specifies # of times to call generator
    autoencoder.fit_generator(autoencoder_generator(train1_path, input_vector_path),
                              samples_per_epoch=model_len, nb_epoch=1)
    fresh_weight_name = weight_folder + weight_name_template + str(num) + weight_name_ext
    autoencoder.save_weights(fresh_weight_name)

# Assuming that all epochs have been processed and no need to check last epoch:
# Each weight file is the next progression (weight update) of the previous weight file
# Loading weights trained from autoencoder
# Encoder can "see" weights because by_name = True meaning that "trained by name" occurred
print('Loading latest weight file into encoder model...')
latest_weight = weight_folder + weight_name_template + str(epoch_num-1) + weight_name_ext
encoder.load_weights(latest_weight, by_name=True)

# Prepare inputs for imdb_regression model:
transformed_data_name = 'transformed_data.h5'
transformed_data_path = parent_path+transformed_data_name
# Use auto_encoder to encode data into small dimension (utilizing encoder layer):
if not os.path.exists(transformed_data_path):
    print('Transforming input data into lower dimensionality...')
    transformed_data = []
    for row in input_data:
        placeholder_input = np.zeros((1, model_size))
        placeholder_input[0] = row
        # Gives result of encoder layer in auto_encoder function
        # Predict doesn't update error - simply multiplies input by weights
        en_predict = encoder.predict(placeholder_input)
        transformed_data.append(en_predict)
    # Save transformed inputs into h5 file:
    print('Saving transformed data into h5py format...')
    transformed_data_name = 'transformed_data'
    transformed_data_file = h5py.File(transformed_data_path, 'wb')
    transformed_data_file.create_dataset(transformed_data_name, data=transformed_data)
else:
    print('Data has already been transformed in dimensionality!')
    print('Loading transformed data...')
    with h5py.File(transformed_data_path, 'r') as load_transformed:
        transformed_data = load_transformed['transformed_data'][:]

##########
# Post Auto-Encoder Portion:
print('Progressing onto regression model...')
# Find length of input data from transformed data:
transformed_len = len(transformed_data[0])
# transformed_size = (1, transformed_len)
transformed_size = (transformed_len, )
# Create model object of imdb regression model:
imdb_reg = imdb_regression(transformed_size)
# Fit generator to model and save weights:
# Load or create weight files for imdb_reg:
# Weight file details:
imdb_weight_folder = parent_path + 'imdb_training_data/'
imdb_weight_name_template = 'imdb_weights'
imdb_weight_name_ext = '.h5'

# Check if weight folder exists and create if it doesn't:
# Typically all training weights for a model will live in one folder
if not os.path.exists(imdb_weight_folder):
    os.mkdir(imdb_weight_folder)

# If the weight folder exists, check how many epochs have been processed:
imdb_counter = -1
if os.path.exists(imdb_weight_folder):
    # Finding the number assigned to most recent epoch
    for epoch in os.listdir(imdb_weight_folder):
        e = epoch.split(imdb_weight_name_template)
        e = e[1].split(imdb_weight_name_ext)
        e = int(e[0])
        if e > imdb_counter:
            imdb_counter = counter
    print('Latest weight file within IMDb regression weight folder is indexed', counter)
else:
    # Start at epoch 0 if no epochs have been processed
    imdb_counter = -1

# Define name of imdb weight file:
imdb_weight_file_name = imdb_weight_folder + imdb_weight_name_template + str(imdb_counter) \
                        + imdb_weight_name_ext
# Load imdb weight file if it exists:
if os.path.exists(imdb_weight_file_name):
    print('Latest imdb weight file already exists! Loading...')
    # by_name allows for old weight files to be used with new models with new structures
    autoencoder.load_weights(weight_file_name, by_name=True)

# Specify maximum number of epochs to process in imdb weight training:
imdb_epoch_num = 10
for num in range(imdb_counter+1, imdb_epoch_num):
    print('IMDb regression weight training on epoch:', num)
    # Train (implies updating error) and generate weights:
    # nb_epoch specifies how many epochs run before saving (1 in this case)
    # samples_per_epoch specifies # of times to call generator
    imdb_reg.fit_generator(input_generator(train1_path, output_vector_path, input_vector_path),
                           samples_per_epoch=transformed_len, nb_epoch=1)
    fresh_imdb_weight_name = imdb_weight_folder + imdb_weight_name_template + str(num) + imdb_weight_name_ext
    imdb_reg.save_weights(fresh_imdb_weight_name)
# Load latest weights for imdb model:
imdb_latest_weight = imdb_weight_folder + imdb_weight_name_template + str(imdb_epoch_num-1) + imdb_weight_name_ext
imdb_reg.load_weights(imdb_latest_weight, by_name=True)

# Run predictions on test data:
# Load test index file and output data file:
with open(test_name_path, 'rb') as test_set:
    test_data = pkl.load(test_set)
with open(output_vector_path, 'rb') as output_set:
    output_data = pkl.load(output_set)

# Specify save parameters for predictions:
prediction_data_name = 'score_predictions.csv'
prediction_data_path = parent_path+prediction_data_name
if not os.path.exists(prediction_data_path):
    print('Running predictions on test data...')
    prediction_vec = []
    score_vec = []
    # Train on test data:
    for index in test_data:
        # Pull necessary data using index in test data:
        test_row = input_data[index]
        score_val = output_data[index]
        # Populate placeholder vector:
        placeholder_input = np.zeros((1, transformed_len))
        placeholder_input[0] = test_row
        imdb_predict = imdb_reg.predict(placeholder_input)
        prediction_vec.append(imdb_predict)
        score_vec.append(score_val)
    print('Predictions have completed! Proceeding to save data...')
    final_results = pd.DataFrame()
    final_results['Real_Score'] = score_vec
    final_results['Predicted_Score'] = prediction_vec
    final_results.to_csv(prediction_data_path)
else:
    print('Predictions for IMDb scores have already been run!')
    final_results = pd.read_csv(prediction_data_path)

# Output final results:
final_results
