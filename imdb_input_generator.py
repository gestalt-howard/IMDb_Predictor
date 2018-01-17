# Import data from CSV into Pandas Data Frame
# Format for feeding into neural network

import pandas as pd
import numpy as np
from gensim.models import Word2Vec  # Latent semantic analysis package
import os


# Function to break up strings with pipes
def break_pipe(string_pipe):
    """
    Receives string as input and outputs ordered list of strings from original string with pipes
    """
    broken_line = string_pipe.split('|')
    return broken_line


# Function to replace pipes with spaces
def replace_pipe(string_pipe):
    """
    Receives string as input and outputs the same string with pipes removed and replaced with whitespace
    """
    edit_line = string_pipe.replace('|', ' ')
    return edit_line


# Function to generate keys for transforming data frame vectors:
def key_create(col_df):
    """
    Receives column from pandas data frame and outputs an ordered list of unique values in that column
    """
    col_list = sorted(list(set(col_df)))
    return col_list


# Function to generate neural network input:
def vector_gen(df_row, key_list_df):
    """
    Receives row from pandas data frame and converts it into required input to neural network
    Conversion is done using keys stored in key_list
    """
    input_list = []
    binary_list = [0, 1, 4, 7, 8, 10, 11, 13, 15]
    norm_list = [2, 3, 5, 12]
    for ind, element in enumerate(df_row):
        if ind in binary_list:
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
            # Normalize using numpy function if possible
            print('Index:', ind)
            norm_val = [normalize_value_generate(element, key_list_df, ind)]
            print(norm_val)
            input_list.append(norm_val)
    # Remove boundaries between individual elements in input list
    final_vector = []
    for entry in input_list:
        final_vector = final_vector + list(entry)
    print(final_vector[1650:1670])
    return final_vector


# Function to generate binary vectors:
def binary_vector_generate(val, lookup_key, ind):
    """
    Receives values and returns a binary vector
    Searches along key to find location of target element and updates a zero-vector with an added 1 as appropriate
    """
    lu_elem = lookup_key[ind]
    vec_gen = np.zeros(len(lu_elem))
    if type(val) == list:
        for val_elem in val:
            vec_gen[lu_elem.index(val_elem)] = 1
    else:
        vec_gen[lu_elem.index(val)] = 1
    return vec_gen


# Function to normalize inputs to 0-100 scale
def normalize_value_generate(val, lookup_key, ind):
    """
    Receives value and returns a normalized value based on lookup key
    """
    lu_elem = lookup_key[ind]
    elem_min = min(lu_elem)
    lu_elem[:] = [x - elem_min for x in lu_elem]
    elem_shifted_max = max(lu_elem)
    norm_val = (float(val-elem_min)/float(elem_shifted_max))*100
    return norm_val


# Set file path parameters:
parent_path = '/Users/cheng-haotai/Documents/Projects_Data/IMDb_Predictor/'
csv_name = 'movie_metadata.csv'
csv_path = parent_path + csv_name
# Set model save path for Word2Vec model
model_name = 'word_model.bin'
model_path = parent_path + model_name

# Read CSV into pandas dataframe:
imdb_df = pd.read_csv(csv_path)

print('Type of imdb_df:')
print(type(imdb_df))
print('Size of data frame:')
print(imdb_df.shape)

# Predefine expected column data types:
dtypes_dict = {'color': str, 'director_name': str, 'num_critic_for_reviews': np.int, 'duration': np.int,
               'director_facebook_likes': np.int, 'actor_3_facebook_likes': np.int, 'actor_2_name': str,
               'actor_1_facebook_likes': np.int, 'gross': np.int, 'genres': str, 'actor_1_name': str,
               'movie_title': str, 'num_voted_users': np.int, 'cast_total_facebook_likes': np.int, 'actor_3_name': str,
               'facenumber_in_poster': np.int, 'plot_keywords': str, 'movie_imdb_link': str,
               'num_user_for_reviews': np.int, 'language': str, 'country': str, 'content_rating': str, 'budget': np.int,
               'title_year': np.int, 'actor_2_facebook_likes': np.int, 'imdb_score': np.float, 'aspect_ratio': np.float,
               'movie_facebook_likes': np.int}

# Clean the data frame of rows with null values:
print('Column with most null values has %s missing values' % (imdb_df.isnull().sum().max()))
print('Remove all rows with any missing values...')
imdb_df.dropna(how='any', inplace=True)
print('Shape of reduced data frame:', imdb_df.shape)

# Specify data types for columns in data frame:
for ent in dtypes_dict:
    imdb_df[str(ent)] = imdb_df[str(ent)].astype(dtypes_dict[ent])
# Clean up string fields of escape characters:
str_df = imdb_df.select_dtypes(include=['object'])
for col in str_df.columns.values:
    imdb_df[str(col)] = imdb_df[str(col)].str.strip()

# Reduce data frame to only desired fields:
desired_fields = ['color', 'director_name', 'num_critic_for_reviews', 'duration', 'actor_2_name', 'gross', 'genres',
                  'actor_1_name', 'actor_3_name', 'plot_keywords', 'country', 'content_rating', 'budget', 'title_year',
                  'imdb_score', 'aspect_ratio']
# Make new data frame that is a subset of imdb_df:
work_df = imdb_df[desired_fields].copy(deep=True)
print('Shape of work data frame:', work_df.shape)

# Generate transformation keys:
key_list = []
# Actor names must be aggregated for key generation:
# Index of actor names (for aggregation purposes)
actor_index = [4, 7, 8]
actor_agg = []
# Piped strings need to be separated:
# Index of genre category (6) and keyword category (9)
genre_index = 6
keyword_index = 9
genre_agg = []
keyword_agg = []
# Generate aggregated list of keys (non-unique) by iterating through columns of data frame
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
print(len(key_list[1]))

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

# Find the number of words in the longest phrase in plot keyword list:
phrase_len = []
for key_phrase in keyword_key:
    phrase_len.append(len(key_phrase))
longest_len = max(phrase_len)

# Generate a model using Word2Vec to capture relative "closeness" of phrases in plot keyword
# Word2Vec produces word vector using deep learning
# Word2Vec creates model from phrases and the model embeds context information into words
# Specify parameters for Word2Vec
vector_size = 300
min_count = 1
alpha = 0.025
if not os.path.exists(model_path):
    print('Training Word2Vec model...')
    word_model = Word2Vec(keyword_key, size=vector_size, min_count=min_count, alpha=alpha, hs=1, negative=0)
    word_model.save(model_path)
else:
    print('Loading trained model...')
    word_model = Word2Vec.load(model_path)

# Generate output vector:
output_vector = list(work_df.imdb_score)

# Generate input vectors row-by-row:
input_vectors = []
print('Assembling input vector...')
for idx, row in work_df.iterrows():
    vector_temp = vector_gen(row, key_list)
    input_vectors.append(vector_temp)
    break

print('Length of input vector is:', len(input_vectors))