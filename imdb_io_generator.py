import sqlite3 as sql
import sys
import os
import numpy as np

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
					# print ("pass1")
					# print (bin_vector)
					# print (len(bin_vector))
				else:
					pos_idx = type_ranges[i].index(db_vector[i])
					bin_vector = [0]*len(type_ranges[i])
					bin_vector[pos_idx] = 1
					input_vector = input_vector + bin_vector
					# print ("pass2")
					# print (bin_vector)
					# print (len(bin_vector))
			else:
				norm_value = float(db_vector[i] - type_ranges[i][0]) / float(type_ranges[i][1] - type_ranges[i][0])
				input_vector = input_vector + [norm_value]
				# print ("pass3")
				# print (norm_value)
				# print ("1")
		else:
			if type(type_ranges[i][0]) is not int and type(type_ranges[i][0]) is not float:
				bin_vector = [0]*len(type_ranges[i])
				if db_vector[i] in type_ranges[i]:
					pos_idx = type_ranges[i].index(db_vector[i])
					bin_vector[pos_idx] = 1
				input_vector = input_vector + bin_vector	
				# print ("pass4")
				# print (bin_vector)
				# print (len(bin_vector))
			else:
				norm_value = -1
				input_vector = input_vector + [norm_value]
				# print ("pass5")
				# print (norm_value)
				# print ("1")
	# print (len(input_vector))
	return input_vector

def scores_to_output_vector_int(score, num_classes, max_score):
	interval = float(max_score)/float(num_classes)
	output_vector = [0]*num_classes
	bin = np.ceil(float(score)/float(interval)) - 1
	bin = bin.astype(int)
	if bin < 0:
		bin = 0
	if bin >= num_classes:
		bin = num_classes - 1
	output_vector[bin] = 1
	return output_vector

def scores_to_output_vector_str(item, type_range):
	pos_idx = type_range.index[item]
	output_vector = [0]*len(type_range)
	output_vector[pos_idx] = 1
	return output_vector
	
if (len(sys.argv) == 4):
	parent_path = sys.argv[1]
	db_name = sys.argv[2]
	output_parameter = sys.argv[3]
else:
	parent_path = "C:/Users/AaronWu/Documents/movie_review_proj/"
	db_name = "movie_metadata.db"
	output_parameter = "imdb_score"
	
save_input_path = parent_path + "movie_input_vectors.npy"
save_output_path = parent_path + "movie_output_vectors.npy"

db_path = parent_path + "/" + db_name

num_classes = 100
max_score = 10
max_items = 100
delimiter = "|"

table_name = db_name[:-3]
db_connection = sql.connect(db_path)
db_cursor = db_connection.cursor()

get_data_cmd = "SELECT * FROM " + table_name
db_cursor.execute(get_data_cmd)
all_data = db_cursor.fetchall()

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

output_type_range = type_ranges[output_param_idx]
del type_ranges[0]
del type_ranges[output_param_idx]
# initial_array = 0
# debug = 1
# iteration = 1
# max_iterations = 2
# debug_item = 4
for a in range(len(all_data)):
	# if debug == 1 and iteration >= max_iterations:
		# break
	# if debug == 1:
		# a = debug_item
		# iteration += 1
	db_vector = list(all_data[a])
	output_value = db_vector[output_param_idx]
	del db_vector[0]
	del db_vector[output_param_idx]
	if a == 0:
		input_vectors = np.array([db_vector_to_input_vector(db_vector, type_ranges, delimiter)])
		if type(output_type_range[0]) is int or type(output_type_range[0]) is float:
			output_vectors = np.array([scores_to_output_vector_int(output_value, num_classes, max_score)])
		else:
			output_vectors = np.array([scores_to_output_vector_str(output_value, output_type_range)])
		initial_array = 1
	else:
		input_vectors = np.concatenate((input_vectors, np.array([db_vector_to_input_vector(db_vector, type_ranges, delimiter)])), axis=0)
		if type(output_type_range[0]) is int or type(output_type_range[0]) is float:
			output_vectors = np.concatenate((output_vectors, np.array([scores_to_output_vector_int(output_value, num_classes, max_score)])), axis=0)
		else:
			output_vectors = np.concatenate((output_vectors, np.array([scores_to_output_vector_str(output_value, output_type_range)])), axis=0)

input_vectors = np.asarray(input_vectors)
np.save(save_input_path, input_vectors)
	
output_vectors = np.asarray(output_vectors)
np.save(save_output_path, output_vectors)