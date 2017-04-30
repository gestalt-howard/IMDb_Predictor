# IMDb_Predictor
Takes movie data from CSV file to predict movies' IMDb score

software prerequisites- 

1. python (preferably 3.5.0)
2. pip install sqlite3 (should be included with python)
3. pip install numpy (should be included with python)
4. pip install pandas
5. install tensorflow (https://www.tensorflow.org/install/)

data prerequisites-

1. csv formatted with label names as first row and columns of values
2. advised that one label is "movie_title" and output label is "imdb_score"
3. advised dataset- https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset

pipeline order-

1. imdb_sql_generator- takes a csv file of movie data and converts each row to a query to be stored in a database

python imdb_sql_generator.py database_directory_path database_name csv_path (arguments optional)

-if choose to have no arguments- go into code and pre-set the arguments:

parent_path = database_directory_path

db_name = database_name

csv_path = csv_path

2. imdb_io_generator- takes data arrays from database and converts them to usable vectors for machine learning (both input and output)- output vectors are binary labels
3. imdb_io_generator_reg- takes data arrays from database and converts them to usable vectors for machine learning (both input and output)- output vectors are single score values

python imdb_io_generator.py database_directory_path database_name output_label (arguments optional)

python imdb_io_generator_reg.py database_directory_path database_name output_label (arguments optional)

-if choose to have no arguments- go into code and pre-set the arguments:

parent_path = database_directory_path

db_name = database_name
output_parameter = output_label

4. imdb_fc_nn- takes input arrays and output labels (binary labels) previously generated and trains and tests a fully connected neural network (only 2 layers)
5. imdb_fc_nn_reg- takes input arrays and output labels (single score values) previously generated and trains and tests a fully connected neural network (only 2 layers)

python imdb_fc_nn.py database_directory_path input_vector_name label_vector_name neural_network_name (arguments optional)

python imdb_fc_nn_reg.py database_directory_path input_vector_name label_vector_name neural_network_name (arguments optional)

-if choose to have no arguments- go into code and pre-set the arguments:

parent_path = database_directory_path

input_name = input_vector_name (movie_input_vectors.npy)

output_name = label_vector_name (movie_output_vectors.npy)

network_name = neural_network_name
	
files generated-

1. imdb_sql_generator-

database- database_directory_path/database_name.db
2. imdb_io_generator-

input vectors- database_directory_path/movie_input_vectors.npy

output vectors- database_directory_path/movie_output_vectors.npy

3. imdb_fc_nn-

neural network model- 

database_directory_path/network_name/checkpoint

database_directory_path/network_name/network_name.data-00000-of-00001

database_directory_path/network_name/network_name.index

database_directory_path/network_name/network_name.meta

predictions (of test data)- 

database_directory_path/network_name/predictions.csv
