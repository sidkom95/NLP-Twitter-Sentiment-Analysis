import pickle
import numpy as np
from paths import *
from keras.preprocessing.text import Tokenizer



def create_glove_embedding():
    '''
    This method returns a dict of the word followed by it's vector from the embeddings created by glove_solution
    '''
    vocab = pickle.load(open(EMBEDDING_PATH + 'my_embeddings/Full_data/vocab.pkl', "rb")) #this file was generated by executing vocab.sh
    embedding = np.load(EMBEDDING_PATH + 'my_embeddings/Full_data/embeddings.npy') # dimension set in the glove_solution file embedding_dim 
    word_embedding = {}
    for key in vocab.keys():
        word_embedding[key] = embedding[vocab.get(key)] # keys in this dict are not encoded 
    return word_embedding

def load_pretrained_embeddings(dimension):
    '''
    This method takes a vector dimension and returns a dict with the word followed by it's vector from the pretrained word embeddings
    '''
    word_embedding = {}
    f = open(EMBEDDING_PATH + 'pretrained_embeddings/glove.twitter.27B.'+ str(dimension) +'d.txt','rb') 
    for line in f:
        values = line.split()
        word_embedding[values[0]] = np.array([float(x) for x in values[1:]]) # keys in this dict are encoded 
    f.close()
    return word_embedding
            
def pretrained_embedding_matrix(dimension):
    #word_embedding = pickle.load(open(EMBEDDING_PATH + 'pretrained_embeddings/pretrained_embedding'+str(dimension)+'d.pkl', "rb"))
    word_embedding = load_pretrained_embeddings(dimension)
    embedding_matrix = np.zeros((vocab_size, dimension))
    for word, i in t.word_index.items():
        embedding_vector = word_embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    return embedding_matrix

def create_input_sequence(X , X_test ,MAX_LENGTH ):
    '''
    This methods creates sequnce of numbers for each tweet where each number correspond to a word. These numbers are obtained with Tokenizer
    '''
    
    t = Tokenizer()
    t.fit_on_texts(X)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_X = t.texts_to_sequences(X)
    encoded_test = t.texts_to_sequences(X_test)
    
    #pad the sequences to have same size of MAX_LENGTH
    sequence_X = pad_sequences(encoded_X, maxlen=MAX_LENGTH, padding='post')
    sequence_test = pad_sequences(encoded_kaggle, maxlen=MAX_LENGTH, padding='post')
    return sequence_X , sequence_test

def prepare_DL_input(X , X_test , dimension , MAX_LENGTH ):
    '''
    This methods prepares the input for the deep learning models . It tronsforms the tweets into sequences of numbers where each
    number represent a word . Each sequence is then padded with 0 so that all tweets have length of 30 
    '''
    sequence_X , sequence_test = create_input_sequence(X , X_test ,MAX_LENGTH )
    embedding_matrix = pretrained_embedding_matrix(dimension)
    s = int(len(X)/2)
    y = np.array(s*[0] + s*[1]) #here we use labes 0 for positive tweets and 1 for negative tweets in order to be able to use relu fct
    return sequence_X , sequence_test , embedding_matrix , y


        
    
    


       
    
   