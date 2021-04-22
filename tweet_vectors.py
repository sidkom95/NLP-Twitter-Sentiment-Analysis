import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tweet_embedding(X_train, X_test, word_embedding , method , dimension):
    '''
    Method that creates an embedding for each tweet . It takes as argument :
      *X_train : training dataset
      *X_test : test dataset
      *word embedding
      *method : a string that specifies which method to use to calculate the embedding of each tweet :
         1) 'tfidf' = uses the tfidf vectorizer to compute the embedding of the tweets
         2) 'average' = takes the average of the words embedding of each word in the tweet 
         3) 'tfidf_average' = takes the average of the words embedding multiplied by their respective tfiidf values in the tweet
      *dimension : the dimension of the word embeddings 
      returns the tweets embeddings for the training and test dataset
    '''
    
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
    if method == 'tfidf' :
        print("data using tfidf vectorizer")
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.fit_transform(X_test)
        return X_train_tfidf , X_test_tfidf
    elif method == 'average' :
        print('data using average of words embedding')
        X_train_average = average_word_vectors(X_train , word_embedding , dimension)
        X_test_average  = average_word_vectors(X_test , word_embedding , dimension)
        return X_train_average , X_test_average
    elif method == 'tfidf_average':
        print('data using average words embedding multiplyed with their respective tfidf values')
        X_train_tfidf_average = tfidf_average_word_vectors(X_train , word_embedding , tfidf_vectorizer , dimension )
        X_test_tfidf_average  = tfidf_average_word_vectors(X_test , word_embedding , tfidf_vectorizer , dimension )
        return X_train_tfidf_average , X_test_tfidf_average

def average_word_vectors(X ,word_embedding , dimension ):
    
    
    error = 0
    avg_word_vectors = np.zeros((len(X), dimension ))
    for i, tweet in enumerate(X):
        
        split_tweet = tweet.split()
        nb_words = 0
        
        for word in split_tweet:
            try:
                avg_word_vectors[i] += word_embedding[word.encode()]
                nb_words += 1

            except KeyError: 
                continue
        if (nb_words != 0):
            avg_word_vectors[i] /= nb_words
        
    return avg_word_vectors

def tfidf_average_word_vectors(X ,word_embedding , tfidf_vectorizer ,dimension ):
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    
    error = 0
    avg_word_vectors = np.zeros((len(X), dimension ))
    for i, tweet in enumerate(X):
        
        split_tweet = tweet.split()
        nb_words = 0
        
        for word in split_tweet:
            try:
                avg_word_vectors[i] += word_embedding[word.encode()] * X_tfidf[i,tfidf_vectorizer.vocabulary_[word]]
                nb_words += 1

            except KeyError: 
                continue
        if (nb_words != 0):
            avg_word_vectors[i] /= nb_words
       
    return avg_word_vectors

