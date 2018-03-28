# Movie Score Predictions
# Training Script
# Import the desired functions from the input generator and neural model scripts:
from imdb_input_generator import *
from imdb_neural_model import *

# Import other required libraries:
import pickle as pkl
import h5py
import os
import pandas as pd
import pdb

# Import Keras:
import keras


# Function Definitions:
# Function for checking if weight folder exists:
def weight_folder_check(subject, mainfolder, subfolders):
    if not os.path.exists(mainfolder):
        print 'Creating folders for %s weights...' % (subject)
        os.mkdir(mainfolder)
        for folder in subfolders:
            print 'Creating folder:', folder
            os.mkdir(folder)
    else:
        print 'Folders for %s training weights already exist' % (subject)


# Function for checking the latest training epoch number:
def epoch_check(count, path, name_template, ext):
    for epoch in os.listdir(path):
        if len(epoch.split(name_template)) > 1:
            print 'Epoch name:', epoch
            e = epoch.split(name_template)
            e = e[1].split(ext)
            e = int(e[0])
            if e > count:
                count = e
    return count


# Function for checking latest epoch number for multiple training sets:
def epoch_mult_check(folders, name_template, name_ext):
    ep_counts = []
    counter = -1
    for idx, path in enumerate(folders):
        train_idx = 't%s_' % (idx + 1)
        weight_name_mod = name_template + train_idx
        if os.listdir(path) != []:
            counter = epoch_check(counter, path, weight_name_mod, name_ext)
            print 'Most up-to-date auto-encoder weight file (epoch) for training set %s is indexed: %s\n' % (idx+1,
                                                                                                           counter)
            ep_counts.append(counter)
        else:
            print 'No weight files have been generated yet for training set %s' % (idx+1)
            ep_counts.append(counter)
    return ep_counts


# Defining Folder and File Structure:
# **IMPORTANT: Please remember to update this below cell to whatever main project directory / training data directory structure you've chosen**
# Specify directory containing data files:
parent_path = '/Users/cheng-haotai/Documents/Projects_Data/IMDb_Predictor/'
data_name = 'training_data/'
data_path = parent_path + data_name

# The cells below specify file names and directory structures in relation to the parent path defined above. They do not need to be altered.
# Specify training data files that have been saved:
train1_name = 'train1_index.pickle'
train2_name = 'train2_index.pickle'
train3_name = 'train3_index.pickle'
train1_path = data_path + train1_name
train2_path = data_path + train2_name
train3_path = data_path + train3_name
tpaths = [train1_path, train2_path, train3_path]

train1_folder = 'train1/'
train2_folder = 'train2/'
train3_folder = 'train3/'

# Specify input data file that has been saved:
input_vector_name = 'input_vectors.h5'
input_vector_path = data_path + input_vector_name

# Specify output data file that has been saved:
output_vector_name = 'output_data.pickle'
output_vector_path = data_path + output_vector_name

# Specify test index data files that have been saved:
test_name = 'test_index.pickle'
test_name_path = data_path + test_name

# Prepare inputs for imdb_regression model:
transformed_data_folder = 'transformed_data/'
transformed_data_t1 = 'transformed_data_t1.h5'
transformed_data_t2 = 'transformed_data_t2.h5'
transformed_data_t3 = 'transformed_data_t3.h5'
transdata_folder = data_path + transformed_data_folder

transdata_path1 = transdata_folder + transformed_data_t1
transdata_path2 = transdata_folder + transformed_data_t2
transdata_path3 = transdata_folder + transformed_data_t3
trans_paths = [transdata_path1, transdata_path2, transdata_path3]

transdata1 = 'trans_data_1'
transdata2 = 'trans_data_2'
transdata3 = 'trans_data_3'
transformed_data_dict = [transdata1, transdata2, transdata3]

# Specify save parameters for predictions:
prediction_data_name = 'score_predictions.csv'
prediction_data_path = data_path + prediction_data_name

# Training Models and Running Predictions:
# The following code will be partitioned by function. These functions can be turned on / off by modifying the cell below to "1" or "0".

# Setting flags to turn on/off training segments
regression = 1
ae = 1
test_ae = 1
test_regression = 1


# Auto-Encoder Portion:
# Specify training weight file details for auto-encoder:
weight_folder = data_path + 'autoen_training_weights/'
weight_name_template = 'autoen_weights'
weight_name_ext = '.h5'
ae_train1_folder = weight_folder + train1_folder
ae_train2_folder = weight_folder + train2_folder
ae_train3_folder = weight_folder + train3_folder
ae_folders = [ae_train1_folder, ae_train2_folder, ae_train3_folder]

# Get size of neural network input:
with h5py.File(input_vector_path, 'r') as input_file:
    input_data = input_file['input_dataset'][:]
model_len = len(input_data)
row_len = len(input_data[0])
model_size = (model_len,)  # Tuple of size 1
row_size = (row_len,)

# Check if weight folder exists and create if it doesn't:
weight_folder_check('auto-encoder', weight_folder, ae_folders)

# Check if reduced-dimension dataset folder exists and create if it doesn't:
if not os.path.exists(transdata_folder):
    print 'Creating folder for reduced-dimensionality datasets...'
    os.mkdir(transdata_folder)
else:
    print 'Folder for reduced-dimensionality datasets already exist'

# Check how many epochs have been processed:
# Counter = -1 if no weights have been generated yet
epoch_counts = epoch_mult_check(ae_folders, weight_name_template, weight_name_ext)
print epoch_counts

# Create a model object of the auto-encoder:
print 'Instantiating model for encoder and auto-encoder'
encoder, autoencoder = auto_encoder(row_size)


if test_ae == 1:
    print 'Generating auto-encoder weights...'
    # Specify maximum number of epochs to process:
    epoch_num = 3
    for idx, path in enumerate(tpaths):
        print 'Training auto-encoder weights on training set %s' % (idx + 1)
        train_idx = 't%s_' % (idx + 1)
        weight_name_mod = weight_name_template + train_idx
        for num in range(epoch_counts[idx] + 1, epoch_num):
            print 'Auto-encoder weight generating on epoch', num
            # nb_epoch specifies how many epochs run before saving
            # samples_per_epoch specifies # of times to call generator
            autoencoder.fit_generator(autoencoder_generator(path, input_vector_path),
                                      samples_per_epoch=model_len, nb_epoch=1)
            fresh_weight_name = ae_folders[idx] + weight_name_mod + str(num) + weight_name_ext
            autoencoder.save_weights(fresh_weight_name)
        print 'Weight generation for auto-encoder complete on training set %s\n' % (idx + 1)
else:
    print 'No further auto-encoder weight generation required'

# Check how many epochs have been processed:
epoch_counts = epoch_mult_check(ae_folders, weight_name_template, weight_name_ext)

if ae == 1:
    for idx, folder in enumerate(ae_folders):
        train_idx = 't%s_' % (idx + 1)
        weight_name_mod = weight_name_template + train_idx
        # Loading weights trained from autoencoder | Encoder can "see" weights because by_name = True
        print 'Loading latest weight file for training set %s into encoder model...' % (idx + 1)
        latest_weight = folder + weight_name_mod + str(epoch_counts[idx]) + weight_name_ext
        encoder.load_weights(latest_weight, by_name=True)

        # Use auto_encoder to encode data into small dimension (utilizing encoder layer):
        if not os.path.exists(trans_paths[idx]):
            print('Transforming input data into lower dimensionality...')
            transformed_data = []
            for row in input_data:
                placeholder_input = np.zeros((1, row_len))  # 22000 size placeholder
                placeholder_input[0] = row
                # Gives result of encoder layer in auto_encoder function
                en_predict = encoder.predict(placeholder_input)[0]  # list with a list
                transformed_data.append(en_predict)
            # Save transformed inputs into h5 file:
            print 'Saving transformed data into h5py format...'
            transformed_data_file = h5py.File(trans_paths[idx], 'w')
            transformed_data_file.create_dataset(transformed_data_dict[idx], data=transformed_data)
            print 'Transformed data has been successfully saved for training set %s!\n' % (idx + 1)
        else:
            print 'Data has already been transformed in dimensionality for training set %s!' % (idx + 1)


# Regression Portion:
# Specify training weight file details for imdb regression model:
imdb_weight_folder = data_path + 'imdb_training_data/'
imdb_weight_name_template = 'imdb_weights'
imdb_weight_name_ext = '.h5'
imdb_train1_folder = imdb_weight_folder + train1_folder
imdb_train2_folder = imdb_weight_folder + train2_folder
imdb_train3_folder = imdb_weight_folder + train3_folder
imdb_folders = [imdb_train1_folder, imdb_train2_folder, imdb_train3_folder]

# List out paths of transformed data:
trans_paths

# Get size of transformed data:
print 'Loading transformed data...'
loaded_transdata = []
for idx, path in enumerate(trans_paths):
    with h5py.File(path, 'r') as load_transformed:
            loaded_transdata.append(load_transformed[transformed_data_dict[idx]][:])
# Initialize size values as zeros:
transformed_len = 0
trans_row_len = 0
trans_row_size = 0
for data in loaded_transdata:
    transformed_len += len(data)
    trans_row_len += len(data[0])
# Get 'averaged' data size:
transformed_len = transformed_len / len(loaded_transdata)
trans_row_len = trans_row_len / len(loaded_transdata)
trans_row_size = (trans_row_len,)  # Length 1 tuple (10000 value)
print 'Length of transformed dataset:', transformed_len
print 'Lenght of a row of the transformed dataset', trans_row_len

# Check if weight folder exists and create if it doesn't:
weight_folder_check('IMDb regression', imdb_weight_folder, imdb_folders)

# Check how many epochs have been processed:
# Counter = -1 if no weights have been generated yet
imdb_epoch_counts = epoch_mult_check(imdb_folders, imdb_weight_name_template, weight_name_ext)
print imdb_epoch_counts

# Create model object of imdb regression model:
imdb_reg = imdb_regression(trans_row_size)

if test_regression == 1:
    print 'Generating imdb-regression weights...'
    # Specify maximum number of epochs to process in imdb weight training:
    imdb_epoch_num = 3
    for idx, path in enumerate(tpaths):
        print 'Training imdb-regression weights on training set %s' % (idx + 1)
        train_idx = 't%s_' % (idx + 1)
        weight_name_mod = imdb_weight_name_template + train_idx
        for num in range(imdb_epoch_counts[idx] + 1, imdb_epoch_num):
            print 'IMDb regression weight training on epoch:', num
            imdb_reg.fit_generator(input_generator(path, output_vector_path, trans_paths[idx],
                                                   transformed_data_dict[idx]), samples_per_epoch = transformed_len,
                                                   nb_epoch=1)
            fresh_imdb_weight_name = imdb_folders[idx] + weight_name_mod + str(num) + imdb_weight_name_ext
            imdb_reg.save_weights(fresh_imdb_weight_name)
        print 'Weight generation for imdb-regression complete on training set %s\n' % (idx + 1)
else:
    print 'No further imdb-regression weight generation required'

# Check how many epochs have been processed:
imdb_epoch_counts = epoch_mult_check(imdb_folders, imdb_weight_name_template, weight_name_ext)

# Create references to imdb training weight files:
print 'Trained imdb weight files\' names:'
imdb_trained_weights = []
for idx, epnum in enumerate(imdb_epoch_counts):
    ep_identifier = 't%s_%s' % (idx + 1, epnum)
    file_name = imdb_folders[idx] + imdb_weight_name_template + ep_identifier + imdb_weight_name_ext
    print file_name
    imdb_trained_weights.append(file_name)

# Create folder for imdb predictions:
pred_folder_name = 'movie_predictions/'
pred_folder = data_path + pred_folder_name
if not os.path.exists(pred_folder):
    print 'Creating movie predictions folder...'
    os.mkdir(pred_folder)
else:
    print 'Movie predictions folder already exists'

# Define movie predictions' names
print 'Movie predictions file names:'
pred_names = []
movie_pred_name = 'movie_prediction_'
for i in range(len(imdb_epoch_counts)):
    pred = pred_folder + movie_pred_name + str(i + 1) + '.csv'
    print pred
    pred_names.append(pred)

# Load 'test index' file and 'output data' file:
with open(test_name_path, 'rb') as test_set:
    test_data = pkl.load(test_set)
with open(output_vector_path, 'rb') as output_set:
    output_data = pkl.load(output_set)

if regression == 1:
    for idx, fx in enumerate(imdb_trained_weights):
        # by_name allows for old weight files to be used with new models with new structures
        imdb_reg.load_weights(fx, by_name=True)
        # Load transformed data
        with h5py.File(trans_paths[idx], 'r') as trans_open:
            transformed_data = trans_open[transformed_data_dict[idx]][:]
        # Run predictions on test data:
        print 'Running predictions on test data with training set %s weights...' % (idx + 1)
        prediction_vec = []
        actual_score_vec = []
        for index in test_data:
            # Pull necessary data using index in test data:
            test_row = transformed_data[index]
            score_val = output_data[index]
            # Populate placeholder vector:
            placeholder_input = np.zeros((1, trans_row_len))  # 10000 placeholder
            placeholder_input[0] = test_row
            # Get prediction
            imdb_predict = imdb_reg.predict(placeholder_input)
            prediction_vec.append(imdb_predict)
            actual_score_vec.append(score_val)
        print 'Predictions have completed for training set %s weights! Proceeding to save data...' % (idx + 1)
        final_results = pd.DataFrame()
        final_results['Real_Score'] = actual_score_vec
        final_results['Predicted_Score'] = prediction_vec
        final_results.to_csv(pred_names[idx])
        print 'Prediction data for dataset %s has been saved!\n' % (idx + 1)
else:
    print 'Predictions for IMDb scores have already been run!'
