
# coding: utf-8

# # Movie Score Predictions

# ## Training Script

# The purpose of this script is to tie together the input generator and neural model scripts to train a neural network.

# ### Import Scripts and Libraries:

# In[ ]:


# Import the desired functions from the input generator and neural model scripts:
from imdb_input_generator import *
from imdb_neural_model import *


# In[ ]:


# Import other required libraries:
import pickle as pkl
import h5py
import os
import pandas as pd
import pdb


# In[ ]:


# Import Keras:
import keras


# ### Function Definitions: 

# In[ ]:


# Function for checking if weight folder exists (when multiple training sets are used):
def weight_folder_check(subject, mainfolder, subfolders):
    if not os.path.exists(mainfolder):
        print 'Creating folders for %s weights...' % (subject)
        os.mkdir(mainfolder)
        for folder in subfolders:
            print 'Creating folder:', folder
            os.mkdir(folder)
    else:
        print 'Folders for %s training weights already exist' % (subject)


# In[ ]:


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


# In[ ]:


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


# ### Defining Folder and File Structure:

# **IMPORTANT: Please remember to update this below cell to whatever main project directory / training data directory structure you've chosen**

# In[ ]:


# Specify directory containing data files:
parent_path = '/Users/cheng-haotai/Documents/Projects_Data/IMDb_Predictor/'
data_name = 'training_data/'
data_path = parent_path + data_name


# The cells below specify file names and directory structures in relation to the parent path defined above. They do not need to be altered.

# In[ ]:


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


# In[ ]:


# Specify input data file that has been saved:
input_vector_name = 'input_vectors.h5'
input_vector_path = data_path + input_vector_name


# In[ ]:


# Specify output data file that has been saved:
output_vector_name = 'output_data.pickle'
output_vector_path = data_path + output_vector_name


# In[ ]:


# Specify test index data files that have been saved:
test_name = 'test_index.pickle'
test_name_path = data_path + test_name


# In[ ]:


# Specify details for reduced-dimensionality dataset:
transformed_data_folder = 'transformed_data/'
transformed_data_name = 'transformed_data.h5'
transdata_folder = data_path + transformed_data_folder
transdata_path = transdata_folder + transformed_data_name

transformed_data_dict = 'trans_data'


# ### Training Models and Running Predictions:

# The following code will be partitioned by function. These functions can be turned on / off by modifying the cell below to "1" or "0".

# In[ ]:


# Setting flags to turn on/off training segments 
regression = 1
ae = 1
test_ae = 1
test_regression = 1


# #### Auto-Encoder Portion:

# In[ ]:


# Specify training weight file details for auto-encoder:
ae_weight_folder = data_path + 'autoen_training_weights/'
ae_weight_name_template = 'autoen_weights_'
weight_name_ext = '.h5'


# In[ ]:


# Get size of neural network input:
with h5py.File(input_vector_path, 'r') as input_file:
    input_data = input_file['input_dataset'][:]
model_len = len(input_data)
row_len = len(input_data[0])
model_size = (model_len,)  # Tuple of size 1
row_size = (row_len,)


# In[ ]:


# Check if weight folder exists and create if it doesn't:
if not os.path.exists(ae_weight_folder):
    print 'Making auto-encoder weights folder...'
    os.mkdir(ae_weight_folder)
else:
    print 'Auto-encoder weights folder already exists'


# In[ ]:


# Check if reduced-dimension dataset folder exists and create if it doesn't:
if not os.path.exists(transdata_folder):
    print 'Creating folder for reduced-dimensionality datasets...'
    os.mkdir(transdata_folder)
else:
    print 'Folder for reduced-dimensionality datasets already exists'


# In[ ]:


# Check how many epochs have been processed:
# Counter = -1 if no weights have been generated yet
ae_count = -1
ae_epoch_counts = epoch_check(ae_count, ae_weight_folder, ae_weight_name_template, weight_name_ext)
print 'The latest auto-encoder epoch is indexed at %s' % (ae_epoch_counts)


# In[ ]:


# Create a model object of the auto-encoder:
print 'Instantiating model for encoder and auto-encoder'
encoder, autoencoder = auto_encoder(row_size)


# The preceeding couple of cells under the auto-encoder portion have created the necessary folder structures and collected the relevant information necessary to train the auto-encoder model and reduce dataset dimensionality. Now, the auto-encoder training will begin. Once weights from auto-encoder training have been generated, the most recent weight file will be loaded into the encoder model for dimensionality reduction.

# In[ ]:


if test_ae == 1:
    print 'Generating auto-encoder weights...'
    
    # Specify maximum number of epochs to process:
    epoch_num = 3
    print 'Training auto-encoder weights...'
    for num in range(ae_epoch_counts + 1, epoch_num):
        print 'Auto-encoder weight generating on epoch', num
        # nb_epoch specifies how many epochs run before saving
        # samples_per_epoch specifies # of times to call generator
        autoencoder.fit_generator(autoencoder_generator(input_vector_path), 
                                  samples_per_epoch=model_len, nb_epoch=1)
        fresh_weight_name = ae_weight_folder + ae_weight_name_template + str(num) + weight_name_ext
        autoencoder.save_weights(fresh_weight_name)
    print 'Weight generation for auto-encoder complete!\n'
else:
    print 'No further auto-encoder weight generation required'


# In[ ]:


# Check how many epochs have been processed:
ae_epoch_counts = epoch_check(ae_count, ae_weight_folder, ae_weight_name_template, weight_name_ext)
print 'The latest auto-encoder epoch is indexed at %s' % (ae_epoch_counts)


# In[ ]:


if ae == 1:
    # Loading weights trained from autoencoder | Encoder can "see" weights because by_name = True
    print 'Loading latest auto-enocder weight file for loading into encoder model...'
    latest_weight = ae_weight_folder + ae_weight_name_template + str(ae_epoch_counts) + weight_name_ext
    encoder.load_weights(latest_weight, by_name=True)

    # Use auto_encoder to encode data into small dimension (utilizing encoder layer):
    if not os.path.exists(transdata_path):
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
        transformed_data_file = h5py.File(transdata_path, 'w')
        transformed_data_file.create_dataset(transformed_data_dict, data=transformed_data)
        print 'Transformed data has been successfully saved!\n'
    else:
        print 'Data has already been transformed in dimensionality!'


# #### Regression Portion:

# In[ ]:


# Specify training weight file details for imdb regression model:
imdb_weight_folder = data_path + 'imdb_training_data/'
imdb_weight_name_template = 'imdb_weights_'
imdb_weight_name_ext = '.h5'
imdb_train1_folder = imdb_weight_folder + train1_folder
imdb_train2_folder = imdb_weight_folder + train2_folder
imdb_train3_folder = imdb_weight_folder + train3_folder
imdb_folders = [imdb_train1_folder, imdb_train2_folder, imdb_train3_folder]


# In[ ]:


# Get size of transformed data:
print 'Loading transformed data...'
with h5py.File(transdata_path, 'r') as load_transformed:
    loaded_transdata = load_transformed[transformed_data_dict][:]
        
train_idx_list = []
for train_name in tpaths:
    with open(train_name, 'rb') as pkl_open:
        train_idx_list.append(pkl.load(pkl_open))

# Initialize size values as zeros:
transformed_len = 0
for idx_list in train_idx_list:
    transformed_len += len(idx_list)
# Get 'averaged' data size:
transformed_len = transformed_len / len(train_idx_list)
    
trans_row_len = len(loaded_transdata[0])
trans_row_size = (trans_row_len,)  # Length 1 tuple (10000 value)

print 'Length of input dataset into imdb_regression model:', transformed_len
print 'Lenght of a row of the transformed dataset', trans_row_len


# In[ ]:


# Check if weight folder exists and create if it doesn't:
weight_folder_check('IMDb regression', imdb_weight_folder, imdb_folders)


# In[ ]:


# Check how many epochs have been processed:
# Counter = -1 if no weights have been generated yet
imdb_epoch_counts = epoch_mult_check(imdb_folders, imdb_weight_name_template, weight_name_ext)
print imdb_epoch_counts


# In[ ]:


# Create model object of imdb regression model:
imdb_reg = imdb_regression(trans_row_size)


# In[ ]:


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
            imdb_reg.fit_generator(input_generator(path, output_vector_path, transdata_path, 
                                                   transformed_data_dict), samples_per_epoch = transformed_len, 
                                                   nb_epoch=1)
            fresh_imdb_weight_name = imdb_folders[idx] + weight_name_mod + str(num) + imdb_weight_name_ext
            imdb_reg.save_weights(fresh_imdb_weight_name)
        print 'Weight generation for imdb-regression complete on training set %s\n' % (idx + 1)
else:
    print 'No further imdb-regression weight generation required'


# In[ ]:


# Check how many epochs have been processed:
imdb_epoch_counts = epoch_mult_check(imdb_folders, imdb_weight_name_template, weight_name_ext)


# In[ ]:


# Create references to imdb training weight files based on latest trained epoch:
print 'Trained imdb weight files\' names:'
imdb_trained_weights = []
for idx, epnum in enumerate(imdb_epoch_counts):
    ep_identifier = 't%s_%s' % (idx + 1, epnum)
    file_name = imdb_folders[idx] + imdb_weight_name_template + ep_identifier + imdb_weight_name_ext
    print file_name
    imdb_trained_weights.append(file_name)


# In[ ]:


# Create folder for imdb predictions:
prediction_path = parent_path + 'prediction_results/'
if not os.path.exists(prediction_path):
    print 'Creating folder for storing prediction values...'
    os.mkdir(prediction_path)
else:
    print 'Prediction values folder already exists!'


# In[ ]:


# Define movie predictions' names
print 'Movie predictions file names:'
pred_names = []
movie_pred_name = 'movie_prediction_'
for i in range(len(imdb_epoch_counts)):
    pred = prediction_path + movie_pred_name + str(i + 1) + '.csv'
    print pred
    pred_names.append(pred)


# In[ ]:


# Load 'test index' file and 'output data' file:
with open(test_name_path, 'rb') as test_set:
    test_data = pkl.load(test_set)
with open(output_vector_path, 'rb') as output_set:
    output_data = pkl.load(output_set)


# In[ ]:


if regression == 1:
    for idx, fx in enumerate(imdb_trained_weights):
        # by_name allows for old weight files to be used with new models with new structures
        print 'Loading weight file %s' % fx
        imdb_reg.load_weights(fx, by_name=True)
        # Load transformed data
        with h5py.File(transdata_path, 'r') as trans_open:
            transformed_data = trans_open[transformed_data_dict][:]
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

