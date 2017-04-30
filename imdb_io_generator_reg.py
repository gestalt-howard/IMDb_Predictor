import sqlite3 as sql
import sys
import os
import numpy as np

#converts the items from the database to a purely numerical array that can be used by neural networks
#strings can be converted to unique binary vectors based on the number of unique strings
#numerical values remain the same
#all of the converted items are concatenated into a new array
#db_vector = vector of items from the database
#type_ranges = array detailing number of unique items for a given feature
#delimiter = a separator in case an feature contains multiple values per item
def db_vector_to_input_vector(db_vector, type_ranges, delimiter):
	input_vector = []
	for i in range(len(db_vector)):			
		if type_ranges[i] == -1:
			continue
		elif db_vector[i] is not None:
			if type(db_vector[i]) is not int and type(db_vector[i]) is not float:
				if delimiter in db_vector[i]:
					bin_vector = [0]*len(type_ranges[i])
					split_vector = db_vector[i].split(delimiter)
					for split_item in split_vector:
						pos_idx = type_ranges[i].index(split_item)
						bin_vector[pos_idx] = 1
					input_vector = input_vector + bin_vector
				else:
					pos_idx = type_ranges[i].index(db_vector[i])
					bin_vector = [0]*len(type_ranges[i])
					bin_vector[pos_idx] = 1
					input_vector = input_vector + bin_vector
			else:
				norm_value = float(db_vector[i] - type_ranges[i][0]) / float(type_ranges[i][1] - type_ranges[i][0])
				input_vector = input_vector + [norm_value]
		else:
			if type(type_ranges[i][0]) is not int and type(type_ranges[i][0]) is not float:
				bin_vector = [0]*len(type_ranges[i])
				if db_vector[i] in type_ranges[i]:
					pos_idx = type_ranges[i].index(db_vector[i])
					bin_vector[pos_idx] = 1
				input_vector = input_vector + bin_vector	
			else:
				norm_value = -1
				input_vector = input_vector + [norm_value]
	return input_vector

#merely converts the score into a list with a single item for compatibility issuse with numpy arrays
def scores_to_output_vector(score):
	output_vector = [score]
	return output_vector
	
#set a path to parent folder that holds the database, the database name, and target feature to use as output
#creates name for input and output numpy files	
if (len(sys.argv) == 4):
	parent_path = sys.argv[1]
	db_name = sys.argv[2]
	output_parameter = sys.argv[3]
else:
	parent_path = "C:/Documents/movie_review_proj/"
	db_name = "movie_metadata.db"
	output_parameter = "imdb_score"
	
save_input_path = parent_path + "movie_input_vectors_reg.npy"
save_output_path = parent_path + "movie_output_vectors_reg.npy"

db_path = parent_path + "/" + db_name

#set number of labels of output and max score of output
#set max items (to filter out string features with too many unique items)
#set delimiter (to parse strings with mulitple values)
num_classes = 100
max_score = 10
max_items = 100
delimiter = "|"

#connect to database
table_name = db_name[:-3]
db_connection = sql.connect(db_path)
db_cursor = db_connection.cursor()

#pull data from database
get_data_cmd = "SELECT * FROM " + table_name
db_cursor.execute(get_data_cmd)
all_data = db_cursor.fetchall()

#get the names of the feature labels themselves
parameter_names = []
get_parameters_cmd = "PRAGMA table_info(" + table_name + ")"
db_cursor.execute(get_parameters_cmd)
header_data = db_cursor.fetchall()
output_param_idx = -1
for header_idx in range(len(header_data)):
	parameter_names = parameter_names + [header_data[header_idx][1]]
	if header_data[header_idx][1] == "imdb_score":
		score_idx = header_idx
	if header_data[header_idx][1] == output_parameter:
		output_param_idx = header_idx

if output_param_idx == -1:
	print ("Given parameter " + output_parameter + " doesn't exist")
	output_param_idx = score_idx
	
#find the ranges of values every feature has- number of unique items for strings, range of numbers for integers
type_ranges = []
has_delimiter = 0
for a in range(len(all_data[0])):
	get_data_cmd = "SELECT " + parameter_names[a] + " FROM " + table_name
	db_cursor.execute(get_data_cmd)
	target_items = db_cursor.fetchall()
	target_items_set = [x[0] for x in list(set(target_items))]
	if type(all_data[0][a]) != int and type(all_data[0][a]) != float:
		if len(target_items_set) < max_items:
			for idx in target_items_set:
				if idx is not None:
					if delimiter in idx:
						has_delimiter = 1
						break
			if has_delimiter == 1:
				total_target_items = []
				for idx in target_items_set:
					if idx is not None:
						total_target_items = total_target_items + idx.split(delimiter)
				target_items_set = list(set(total_target_items))	
		else:
			target_items_set = -1
		type_ranges = type_ranges + [target_items_set]
	else:
		target_items_set = [x for x in target_items_set if x is not None]
		min_max = [min(target_items_set), max(target_items_set)]
		type_ranges = type_ranges + [min_max]

#build the input and output vectors to be saved
output_type_range = type_ranges[output_param_idx]
del type_ranges[0]
del type_ranges[output_param_idx]
for a in range(len(all_data)):
	db_vector = list(all_data[a])
	output_value = db_vector[output_param_idx]
	del db_vector[0]
	del db_vector[output_param_idx]
	if a == 0:
		input_vectors = np.array([db_vector_to_input_vector(db_vector, type_ranges, delimiter)])
		output_vectors = np.array([scores_to_output_vector(output_value)])
	else:
		input_vectors = np.concatenate((input_vectors, np.array([db_vector_to_input_vector(db_vector, type_ranges, delimiter)])), axis=0)
		output_vectors = np.concatenate((output_vectors, np.array([scores_to_output_vector(output_value)])), axis=0)

#save input and output vectors
input_vectors = np.asarray(input_vectors)
np.save(save_input_path, input_vectors)
	
output_vectors = np.asarray(output_vectors)
np.save(save_output_path, output_vectors)