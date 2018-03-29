
# coding: utf-8

# # Input Collector for IMDb Predictor Neural Network

# The purpose of this script is to extract data from a CSV file containing information on approximately 5000 movies, clean the extracted data, and format it for input into a neural network.

# In[ ]:


# Import necessary libraries:
from gensim.models import Word2Vec  # Latent semantic analysis package
from os.path import exists as ex

import pandas as pd
import numpy as np
import os
import pickle as pkl
import h5py


# Define functions:

# In[ ]:


# Function to extract index of desired fields:
def target_field(master_list, desired_list):
    desired_index = []
    for elem in desired_list:
        desired_index.append(master_list.index(elem))
    return desired_index


# In[ ]:


# Function to generate keys for transforming data frame vectors:
def key_create(col_df):
    col_list = sorted(list(set(col_df)))
    return col_list


# In[ ]:


# Function to break up strings with pipes
def break_pipe(string_pipe):
    broken_line = string_pipe.split('|')
    return broken_line


# In[ ]:


# Function to replace pipes with spaces
def replace_pipe(string_pipe):
    edit_line = string_pipe.replace('|', ' ')
    return edit_line


# In[ ]:


# Function to generate binary vectors:
def binary_vector_generate(val, lookup_key, ind):
    lu_elem = lookup_key[ind]
    vec_gen = np.zeros(len(lu_elem), dtype=np.int8)
    if type(val) == list:
        for val_elem in val:
            vec_gen[lu_elem.index(val_elem)] = 1
    else:
        vec_gen[lu_elem.index(val)] = 1
    return vec_gen


# In[ ]:


# Function to normalize inputs to 0-100 scale
def normalize_value_generate(val, lookup_key, ind):
    lu_elem = lookup_key[ind]
    elem_min = min(lu_elem)
    lu_elem[:] = [x - elem_min for x in lu_elem]
    elem_shifted_max = max(lu_elem)
    norm_val = (float(val-elem_min)/float(elem_shifted_max))*100
    return norm_val


# In[ ]:


# Function to generate neural network input:
def vector_gen(df_row, key_list_df, longest_len, vector_size, word_model, desired_fields):
    
    input_list = []
    genre_index = desired_fields.index('genres')
    keyword_index = desired_fields.index('plot_keywords')
    
    binary_names = ['color', 'director_name', 'actor_2_name', 'actor_1_name', 'actor_3_name', 'country', 'content_rating',
                    'title_year', 'aspect_ratio']
    binary_list = target_field(desired_fields, binary_names)
    
    norm_names = ['num_critic_for_reviews', 'duration', 'gross', 'budget']
    norm_list = target_field(desired_fields, norm_names)
    
    # Iterate across elements in a row:
    for ind, element in enumerate(df_row):
        if ind in binary_list:
            # Generate binary vector
            input_list.append(binary_vector_generate(element, key_list_df, ind))
        elif ind == genre_index:
            # Special treatment to split words by pipes
            word_list = break_pipe(element)
            input_list.append(binary_vector_generate(word_list, key_list_df, ind))
        elif ind == keyword_index:
            # Special treatment using word embedding
            vsize = longest_len*vector_size
            word_vec = np.zeros(vsize)
            embed_words = replace_pipe(element)
            embed_words = embed_words.split()
            for index, word in enumerate(embed_words):
                word_vec_temp = word_model[word]
                word_vec[(index*vector_size):((index+1)*vector_size)] = word_vec_temp
            input_list.append(word_vec)
        elif ind in norm_list:
            # Normalize value
            norm_val = [normalize_value_generate(element, key_list_df, ind)]
            input_list.append(norm_val)
    # Remove boundaries between individual elements in input list
    # Before boundary removal, each attribute has an entry in the list and we need a straight aggregation
    final_vector = []
    for entry in input_list:
        final_vector = final_vector + list(entry)
    return final_vector


# Main function:

# In[ ]:


# Define parent path of working directory:
parent_path = '/Users/cheng-haotai/Documents/Projects_Data/IMDb_Predictor/'


# In[ ]:


# Define path of predictions folder (for later analysis):
prediction_path = parent_path + 'prediction_results/'
if not os.path.exists(prediction_path):
    print 'Creating folder for storing prediction values...'
    os.mkdir(prediction_path)
else:
    print 'Prediction values folder already exists!'


# In[ ]:


# Set csv file path parameters:
csv_name = 'movie_metadata.csv'
csv_path = parent_path + csv_name


# In[ ]:


# Set model save path for Word2Vec model
model_name = 'word_model.bin'
model_folder = 'training_data/'
model_path = parent_path + model_folder + model_name


# In[ ]:


print(model_path)


# In[ ]:


# Read CSV into pandas dataframe:
imdb_df = pd.read_csv(csv_path)
print 'Type of imdb_df:', type(imdb_df)
print 'Size of data frame:', imdb_df.shape


# In[ ]:


# Predefine expected column data types:
dtypes_dict = {'color': str, 'director_name': str, 'num_critic_for_reviews': np.int, 'duration': np.int,
               'director_facebook_likes': np.int, 'actor_3_facebook_likes': np.int, 'actor_2_name': str,
               'actor_1_facebook_likes': np.int, 'gross': np.int, 'genres': str, 'actor_1_name': str,
               'movie_title': str, 'num_voted_users': np.int, 'cast_total_facebook_likes': np.int, 'actor_3_name': str,
               'facenumber_in_poster': np.int, 'plot_keywords': str, 'movie_imdb_link': str,
               'num_user_for_reviews': np.int, 'language': str, 'country': str, 'content_rating': str, 'budget': np.int,
               'title_year': np.int, 'actor_2_facebook_likes': np.int, 'imdb_score': np.float, 'aspect_ratio': np.float,
               'movie_facebook_likes': np.int}


# In[ ]:


# Clean the data frame of rows with null values:
print 'Column with most null values has %s missing values' % (imdb_df.isnull().sum().max())
print 'Removing all rows with any missing values...'
imdb_df.dropna(how='any', inplace=True)
print 'Shape of reduced data frame:', imdb_df.shape


# In[ ]:


# Specify data types for columns in data frame:
for ent in dtypes_dict:
    imdb_df[str(ent)] = imdb_df[str(ent)].astype(dtypes_dict[ent])
# Clean up string fields of escape characters:
str_df = imdb_df.select_dtypes(include=['object'])
for col in str_df.columns.values:
    imdb_df[str(col)] = imdb_df[str(col)].str.strip()


# In[ ]:


# Reduce data frame to only desired fields:
desired_fields = ['color', 'director_name', 'num_critic_for_reviews', 'duration', 'actor_2_name', 'gross', 'genres',
                  'actor_1_name', 'actor_3_name', 'plot_keywords', 'country', 'content_rating', 'budget', 'title_year',
                  'imdb_score', 'aspect_ratio']
# Make new data frame that is a subset of imdb_df:
work_df = imdb_df[desired_fields].copy(deep=True)
print 'Shape of work data frame:', work_df.shape


# The general strategy for generating inputs for the neural network will be to either binarize (create vector of 0's and 1's with 1's indicating the presence of an entity) or normalize the values along each row. In the special case of plot keywords, word embedding using Word2Vec will be used to transform the keywords into usable input entities.
# 
# Word2Vec produces a "word vector" using deep learning. Using phrases, Word2Vec generates a model that embeds context information from words / placement of words.
# 
# Below, I will generate a "unique" list of each feature's elements for the purpose of creating either binary vectors or normalized values.

# In[ ]:


# Generate transformation keys:
key_list = []
# Actor names must be aggregated for key generation:
actor_index_name = ['actor_2_name', 'actor_1_name', 'actor_3_name']
actor_index = target_field(desired_fields, actor_index_name)
print 'Column index of actor names:', actor_index
actor_agg = []

# Piped strings need to be separated:
genre_index = desired_fields.index('genres')
keyword_index = desired_fields.index('plot_keywords')
print 'Index of genre:', genre_index
print 'Index of plot keywords:', keyword_index
genre_agg = []
keyword_agg = []


# In[ ]:


# Generate aggregated list of unique values by iterating through columns of data frame
for idx, col in enumerate(desired_fields):
    df_col = work_df[str(col)]
    # Checking for actor name
    if idx in actor_index:
        act_temp = key_create(df_col)
        actor_agg += act_temp
        key_list.append('ACTOR')
    # Checking for genre
    elif idx == genre_index:
        for phrase in df_col:
            phrase_temp = break_pipe(phrase)
            genre_agg += phrase_temp
        key_list.append('GENRE')
    # Checking for plot keyword
    elif idx == keyword_index:
        for phrase in df_col:
            phrase_temp = replace_pipe(phrase)
            phrase_temp = phrase_temp.split()
            keyword_agg.append(phrase_temp)
        key_list.append('KEYWORD')
    else:
        key_temp = key_create(df_col)
        key_list.append(key_temp)


# In[ ]:


# Create keys from aggregated lists
act_key = sorted(list(set(actor_agg)))
genre_key = sorted(list(set(genre_agg)))
keyword_key = sorted(keyword_agg)
# Replace placeholder names in list of keys with aggregated keys
for idx, elem in enumerate(key_list):
    if elem == 'ACTOR':
        key_list[idx] = act_key
    elif elem == 'GENRE':
        key_list[idx] = genre_key
    elif elem == 'KEYWORD':
        key_list[idx] = keyword_key


# In[ ]:


# Find the number of words in the longest phrase in plot keyword list to standardize input length for keywords:
phrase_len = []
for key_phrase in keyword_key:
    phrase_len.append(len(key_phrase))
longest_len = max(phrase_len)
print 'Longest phrase in plot keyword has %s number of words' % (longest_len)


# In[ ]:


# Create folder for storing data:
model_loc = parent_path + model_folder
if not ex(model_loc):
    print 'Generating folder for storing training, test, and validation data...'
    os.mkdir(model_loc)
    print 'Data folder successfully generated!'
else:
    print 'Data folder already exists!'


# In[ ]:


# Generate a model using Word2Vec to capture relative "closeness" of phrases in plot keyword
# Specify parameters for Word2Vec
vector_size = 300
min_count = 1
alpha = 0.025
if not ex(model_path):
    print 'Training Word2Vec model...'
    word_model = Word2Vec(keyword_key, size=vector_size, min_count=min_count, alpha=alpha, hs=1, negative=0)
    word_model.save(model_path)
    print 'Word2Vec model finished generating!'
else:
    print 'Loading trained model...'
    word_model = Word2Vec.load(model_path)
    print 'Word2Vec model finished loading!'


# In[ ]:


# Generate output vector:
output_vector = list(work_df.imdb_score)
output_vector_name = 'output_data.pickle'
output_path = parent_path + model_folder + output_vector_name
if not ex(output_path):
    print('Saving output vector pickle file...')
    with open(output_path, 'wb') as output_out:
        pkl.dump(output_vector, output_out)
    print('Output vector pickle file successfully saved!')
else:
    print('Output vector pickle file already exists!')


# In[ ]:


# Input vector parameters:
input_vector_name = 'input_vectors.h5'
input_vector_path = parent_path + model_folder + input_vector_name
input_dataset_name = 'input_dataset'

if not ex(input_vector_path):
    # Generate input vectors row-by-row:
    input_vectors = []
    print 'Assembling input vector...'
    for idx, row in work_df.iterrows():
        vector_temp = vector_gen(row, key_list, longest_len, vector_size, word_model, desired_fields)
        input_vectors.append(vector_temp)
    print 'Length of input vector is:', len(input_vectors)
    print 'Length of element in input vector:', len(input_vectors[0])
    # Save input vectors:
    print 'Saving input vector h5 file...'
    input_out = h5py.File(input_vector_path, 'w')
    input_out.create_dataset(input_dataset_name, data=input_vectors)
    input_out.close()
    print 'Input vector h5 file successfully saved!'
else:
    print 'Input vector h5 file already exists!'
    # Load data from h5 file:
    print 'Loading input vector h5 file...'
    with h5py.File(input_vector_path, 'r') as input_grab:
        input_vectors = input_grab['input_dataset'][:]
    print 'Successfully loaded input vector h5 file!'


# At this point, I've created an h5 file containing the transformed version of every single row of the original IMDb dataframe. The transformation has summarized each row into a series of numbers that are mixtures of binary vectors, normalized values, and word vectors containing context information.
# 
# Now, we will proceed to split this master list into 3 separate training sets. Each of these training sets will be used to train a neural network model and the results of 3 separate neural networks will be averaged to get final prediction values.
# 
# The split will be accomplished by defining indicies by which to reference the master-list of input data. 

# In[ ]:


# Define master indices list:
shuffled_indices = np.random.permutation(len(input_vectors))
print 'Length of shuffled indices:', len(shuffled_indices)


# In[ ]:


# Define data set names:
test_name = 'test_index.pickle'
train1_name = 'train1_index.pickle'
train2_name = 'train2_index.pickle'
train3_name = 'train3_index.pickle'


# In[ ]:


# Define data set paths:
test_path = parent_path + model_folder + test_name
predres_test_path = prediction_path + test_name
train1_path = parent_path + model_folder + train1_name
train2_path = parent_path + model_folder + train2_name
train3_path = parent_path + model_folder + train3_name

train_list = [train1_path, train2_path, train3_path]


# In[ ]:


# Check if data files already exist and generate if they don't exist:
if not ex(test_path) or not ex(train1_path) or not ex(train2_path) or not ex(train3_path):
    print 'Creating data files for training set indices and test set indices...'
    
    # Create and save test indicies:
    test_num = int(np.ceil(0.15 * len(input_vectors)))
    print 'Number of elements in test set is:', test_num
    test_indices = shuffled_indices[-test_num:]
    shuffled_indices = shuffled_indices[:-test_num]
    # Save in training data folder
    with open(test_path, 'wb') as test_out:
        pkl.dump(test_indices, test_out)
    # Save in predictions result folder
    with open(predres_test_path, 'wb') as predres_out:
        pkl.dump(test_indices, predres_out)
    
    # Create and save training indices
    adjusted_len = len(shuffled_indices) - test_num
    cutoff_fig = int(np.floor(adjusted_len*0.9))
    print 'Number of elements in training sets is:', cutoff_fig
    # Iterate through list of training set names:
    for name in train_list:
        new_shuffled = np.random.permutation(shuffled_indices)
        model_test_indices = new_shuffled[:cutoff_fig]
        with open(name, 'wb') as train_out:
            pkl.dump(model_test_indices, train_out)
    print 'All training set indices and test set indicies have been saved!'
else:
    print 'Data files for training set indices and test set indices already exist!'

