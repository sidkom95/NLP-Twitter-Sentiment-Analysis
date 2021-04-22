# Twitter Sentiment Analysis


### Project Specifications

See [Project Specification](https://github.com/epfml/ML_course/tree/master/projects/project2/project_text_classification) at EPFL [/epfml/ML_course](https://github.com/epfml/ML_course) github page.


Please add the text files : train_neg_full.txt train_pos_full.txt test_data.txt in the datasets folder


### Description

This is a Twitter Sentiment Analysis project, done in the scope of EPFL's Masters Machine Learning course.
Given is a dataset of 2.5 Million tweets, and a test data for predictions with 10,000 test tweets.
This competition was hosted in www.aicrowd.com

### To run the project you need the libraries:

numpy

keras

tensorflow

sklearn

matplotlib

pickle

wordninja

re

ktrain ( for bert model ) 

math

### Folders

BertModel: This folder contains our already trained bert model.

cleaned_datasets : This folder contains the pickle files of the cleaned training and test datasets

datasets : This folder contains the text files containing the tweets

Deep_Learning_Models : This folder contains our best deep learning model 

dictionnaries : This folder contains dictionnaries for slang translation and spelling correction downloaded from internet

http://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz (dict1)

http://luululu.com/tweet/typo-corpus-r1.txt (dict2)

pos_neg_words: This folder contains two text files for positive-words and negative words (downloeaded from the internet)

submission:  contains the submitted csv

embeddings: This folder contains 2 sub-folders:

+ my_embedding 

+ pretrained_embeddings: contains the pickle file of the glove embeddings downloaded from Stanford with 200 dimension only 

### Code and notebooks

helper.py: This script contains methods to read , process the data , create dictionnaries and create the submissions csv

paths.py: It contains only the paths used in our code used almost everywhere 

create_emebedding.py: It creates a word embedding using either stanford pretrained files ( specify the dimension ) or by constructing our own glove using the glove solution file

tweet_vectors.py : It created the embedding of the tweets and their sequences for deeplearning 

run.py : Script to create the csv for our best model by calling run_dl_model()


Download glove.twitter.27B.zip from https://nlp.stanford.edu/projects/glove/?fbclid=IwAR1yRzuBFvrUYngB61tEOLXlYoqTaBjbnzJmxz4TQcSIfh4YFYZaXPIyYfA and extract it in 'pretrained_embeddings' folder under the 'embeddings' folder
Download the train and test data from https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files and extract them in 'train_test_data' folder
run the script run.py

### Preprocess
Several preprocessing techniques were used, majority of them were helpful and improved the score, but some were not.

Some of the most influental techniques used were:

+ Emphisizing with external dictionnaries

+ Removing Punctuation and noisy words/characters

+ Correct spelling and map words to actual "real" ones via multiple methods and dicts..

+ Lemmatizing/Stemming

+ Deal with apostrophes/ hashtags..etc


### Train & Predictions:

We used as word embeddings both the Glove Stanford pretrained ones(gave us better results because of the huge dataset they learned from) 
and our own created ones. We then did the fit and predictions using supervised techniques (logistic regression, SVM..) 
and in a second time deep learning techniques (CNN, RNN, BERT..) which gave far better results.
After fitting the model, we can test it locally via cross validation to see how good it works, 
improve the different parameters depending on the seen results and finally predict the labels on the test data 
and push our predictions to AIcrowd platform to see how good they were.


## Contributors:
+ Amir Ghali
+ Mahmoud Sellami
+ Victoria Adcock
