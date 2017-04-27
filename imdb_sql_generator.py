# SQL script to generate database for imdb values
# import sql wrapper (sqlite3) as a shorter name (sql)
# calling wrapper function-- wrapper.function(args) [ex. sql.connect("example.db")]
import sqlite3 as sql
# import sys (system) to allow for python scripts to take arguments from commandline
import sys
# import os allows for file manipulation on your computer
import os
import pandas as pd
import numpy as np

# set a path to parent folder that holds the database
# if path given ihas no arguments then take default path
if (len(sys.argv) == 4):
	parent_path = sys.argv[1]
	db_name = sys.argv[2]
	csv_path = sys.argv[3]
else:
	parent_path = "C:/Users/AaronWu/Documents/movie_review_proj/"
	db_name = "movie_metadata.db"
	csv_path = parent_path + "movie_metadata.csv"

db_path = parent_path + "/" + db_name
	
if not os.path.exists(parent_path):
	try:
		os.mkdir (parent_path)
	except:
		print ("Can't make directory: " + parent_path)

db_csv = pd.read_csv(csv_path)
attributes = db_csv.columns.values
if (len(sys.argv) == 4):
	attribute = attributes[0]
else:
	attribute = "movie_title"
attribute_list = db_csv[attribute]

attribute_types = []
for a in range(len(db_csv.loc[0])):
	if (type(db_csv.loc[0][a])) == np.float64:
		attribute_types = attribute_types + ["FLOAT(2)"]
	elif (type(db_csv.loc[0][a])) == np.int64:
		attribute_types = attribute_types + ["BIGINT"]
	elif (type(db_csv.loc[0][a])) == str:
		attribute_types = attribute_types + ["TEXT"]
	elif (type(db_csv.loc[0][a])) == int:
		attribute_types = attribute_types + ["BIGINT"]
	elif (type(db_csv.loc[0][a])) == float:
		attribute_types = attribute_types + ["FLOAT(2)"]
	else:
		attribute_types = attribute_types + ["TEXT"]

db_connection = sql.connect(db_path)
db_cursor = db_connection.cursor()
table_name = db_name[:-3]
if os.path.exists(db_path):
	print ("Reconnecting with: " + db_path)
else:
	print ("Generating: " + db_path)
	table_gen_cmd = "CREATE TABLE " + table_name + "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
	for attribute_idx in range(len(attributes)):
		table_gen_cmd = table_gen_cmd + str(attributes[attribute_idx]) + " " + str(attribute_types[attribute_idx]) + ", "
	table_gen_cmd = table_gen_cmd[:-2] + ")"
	print ("table_gen_cmd: " + table_gen_cmd)
	try:
		db_cursor.execute(table_gen_cmd)
	except:
		print ("table name may already exist")

add_subject_cmd = "INSERT INTO " + table_name + " ("
add_subject_val = "VALUES( "
for attribute_idx in range(len(attributes)):
	add_subject_cmd = add_subject_cmd + attributes[attribute_idx] + ", "
	add_subject_val = add_subject_val + "?,"
add_subject_cmd = add_subject_cmd[:-2] + ") "
add_subject_val = add_subject_val[:-1] + ")"	
add_subjects_cmd = add_subject_cmd +add_subject_val


get_titles_cmd = "SELECT " +attribute +" FROM " + table_name
db_cursor.execute(get_titles_cmd)
existing_attribute = db_cursor.fetchall()
existing_attribute = [a[0] for a in existing_attribute]
add_subjects_list = []
for idx in range(len(attribute_list)):
	movie_attribute = db_csv[attribute][idx]
	if type(movie_attribute) is str:
		movie_attribute = movie_attribute.decode('utf-8','ignore').encode("ascii", "ignore")
	if movie_attribute not in existing_attribute:
		add_subject_val_list = []
		for attribute_idx in range(len(attributes)):
			subject_val = db_csv.loc[idx][attribute_idx]
			if type(subject_val) == str:
				subject_val = subject_val.decode('utf-8','ignore').encode("ascii", "ignore")
			if type(subject_val) == np.int64:
				subject_val = subject_val.item()
			add_subject_val_list = add_subject_val_list + [subject_val]
				
		add_subjects_list = add_subjects_list + [add_subject_val_list]

db_cursor.executemany(add_subjects_cmd, add_subjects_list)
db_connection.commit()
db_connection.close()