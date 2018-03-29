# IMDb Predictor
Welcome to my personal project! The IMDb Predictor is a neural network that uses deep learning to be able to predict movies' IMDb scores based on a variety of factors (i.e. budget, actor names, director names, facebook likes, etc...). Please continue reading below to find required software packages and instructions for the correct order in which to run the scripts.

![Image of movie reel](https://github.com/gestalt-howard/IMDb_Predictor/blob/master/images/Movie-Tavern-Blog-Hero-Image.jpg)

***DISCLAIMER 1: This is currently an ongoing project. Please check back frequently for updates!***

***DISCLAIMER 2: I do not own this dataset. The original dataset was pulled from: https://www.kaggle.com/tmdb/tmdb-movie-metadata in early 2017. The dataset has since changed and WILL NOT be compatible with the scripts in this repo.***

## Software Prerequisites:
* python (2.7.14 required)
* pip install numpy
* pip install pandas
* pip install gensim
* pip install keras
* pip install h5py
* pip install jupyter

## Data Prerequisites:
* CSV formatted with label names as first row and columns of values
	* Advised Dataset- https://github.com/gestalt-howard/IMDb_Predictor

## Command-Line Option:
**Please be sure that the Python version is 2.7**
1. Run *imdb_input_collector.py* in command line
2. Run *imdb_training_script.py* in command line

## Detailed Pipeline Order:
### 1. Clean Data and Format for Neural Network:
Use either:
* *imdb_input_collector.py*
* *imdb_input_collector.ipynb* for a Jupyter Notebook interface (recommended)

#### 1.1 Actions Taken:
1. **User Action Required:** Ensure that all target file paths (i.e. file name, folder directory structure, etc...) are correct
2. Takes CSV file contents and cleans the data of rows with null values
3. Takes subset of remaining dataset and formats fields into binary vectors, normalized values, and word vector with embedded context information
	* Word vector embedding performed using Word2Vector from the gensim library

#### 1.2 Folders and Files Generated:
**NOTE:** The three training sets are used for generating three different regression models. The predictions generated form these three regression models will be averaged to yield greater stability.
* ***training_data (folder)***
	* **word_model.bin**
		* Binary file generated by Word2Vector
		* Model that contains contextual data from text phrases
	* **train1_index.pickle**
		* Contains the row indexes that reference the rows of the *input_vectors.h5* to create the first training set
	* **train2_index.pickle**
		* Same functionality as *train1_index.pickle* but for the second training set
	* **train3_index.pickle**
		* Same functionality as *train1_index.pickle* but for the third training set
	* **test_index.pickle**
		* Same functionality as *train1_index.pickle* but for the test set
		* The test set contains indexes that reference the rows of *input_vectors.h5* that construct the test set
	* **output_data.pickle**
		* Contains the actual IMDb scores that correspond to the rows of the *input_vectors.h5* file
	* **input_vectors.h5**
		* Contains rows of transformed information
		* Text and numerical data have been transformed and normalized into a format suitable for feeding into a neural network
* ***prediction_results (folder)***
	* **test_index.pickle**
		* Same file as in the *training_data* folder
		* This occurrence is used for analysis of the prediction results

### 2. Train the Neural Networks and Predict Results:
Use either:
* *imdb_training_script.py*
* *imdb_training_script.ipynb* for a Jupyter Notebook interface (recommended)

#### 2.1 Actions Taken:
1. **User Action Required:** Ensure that all target file paths (i.e. file name, folder directory structure, etc...) are correct
2. Set the flags to either `0` or `1` depending on which segment of the code you wish to run. There are 4 distinct segments:
	* ***test_ae***: begin or continue generating weight files for a specified number of epochs (autoencoder model)
	* ***ae***: load latest epoch's weights and transform data in dimensionality
	* ***test_regression***: begin or continue generating weight files for a specified number of epochs **for each training set** (IMDb regression model)
	* ***regression***: load latest weight file for IMDb regression and use weights to predict IMDb scores from the test set data
3. Default settings are `1` for all values which will run through the entire code

#### 2.2 Folders and Files Generated:

* ***training_data/autoen_training_weights (folder)***
	* **autoen_weights_0.h5**
	* **autoen_weights_1.h5**
	* **autoen_weights_2.h5**
* ***training_data/imdb_training_data (folder)***
	* ***train1 (folder)***
		* **imdb_weights_t1_0.h5**
		* **imdb_weights_t1_0.h5**
		* **imdb_weights_t1_0.h5**
	* Additional folders ***train2*** and ***train3*** contain similar weight files
* ***training_data/transformed_data (folder)***
	* **transformed_data.h5**
		* This file contains the reduced-dimensionality data that is the result of feeding the rows of *input_vectors.h5* into the autoencoder neural network
* ***prediction_results (folder)***
	* **movie_prediction_1.csv**
		* Contains the predicted and actual IMDb scores generated from the regression model trained from training set 1
	* **movie_prediction_2.csv**
	* **movie_prediction_3.csv**
	* **test_index.pickle**

### 3. Analysis is Performed:
For this section, use only *imdb_analysis.ipynb*. Due to the need for data visualization, Jupyter Notebook is the ideal way for this segment of the project to be presented. Plus, Jupyter is lots of fun!

#### 3.1 Actions Taken:


***TO BE CONTINUED...***
