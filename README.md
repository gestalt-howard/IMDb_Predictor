# IMDb_Predictor
Takes movie data from CSV file to predict movies' IMDb score.
![Image of movie reel](https://github.com/gestalt-howard/IMDb_Predictor/blob/master/images/Movie-Tavern-Blog-Hero-Image.jpg)

***IMPORTANT: This is currently an ongoing project. Please check back frequently for updates!***

***DISCLAIMER***
I do not own this dataset. The original dataset was pulled from: https://www.kaggle.com/tmdb/tmdb-movie-metadata in early 2017. The dataset has since changed and WILL NOT be compatible with the scripts in this repo.

## Software Prerequisites:
* python (2.7.14 required)
* pip install numpy
* pip install pandas
* pip install gensim
* pip install keras
* pip install h5py

## Data Prerequisites:
* CSV formatted with label names as first row and columns of values
	* Advised Dataset- https://github.com/gestalt-howard/IMDb_Predictor

## Quick-and-Dirty Pipeline Order:
**Please be sure that the Python version is 2.7**
1. Run *imdb_input_collector.py* in command line
2. Run *imdb_training_script.py* in command line

## Detailed Pipeline Order:
### 1. Clean Data and Format for Neural Network:
Use either:
* *imdb_input_collector.py*
* *imdb_input_collector.ipynb* for a Jupyter Notebook interface

#### 1.1 Steps:
1. **User Action Required:** Ensure that all target file paths (i.e. file name, folder directory structure, etc...) are correct
2. Takes CSV file contents and cleans the data of rows with null values
3. Takes subset of remaining dataset and formats fields into binary vectors, normalized values, and word vector with embedded context information
	* Word vector embedding performed using Word2Vector from the gensim library


#### 1.2 Files Generated:
* word_model.bin-
	* Binary file generated by Word2Vector
	* Model that contains contextual data from training phrases
	* Model is generated by *imdb_input_generator* and also referenced by *imdb_input_generator* to embed contextual information into word vectors

***TO BE CONTINUED...***
