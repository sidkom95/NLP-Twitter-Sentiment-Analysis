import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import wordninja
from nltk.corpus import words
from paths import *
import itertools
import csv



dict0 = {} # this dictionnary contains the spelling correction collected from 2 dictionnaries found on the internet.
slang = {} # slang dictionnary contains the slang words we will transform them to their original form ,i.e 4u = for you 


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def dict_read(dict_path , dict0) :
    '''
    Creates a dictionnary from a text file
    '''
    dict1 = open(dict_path, 'rb')
    for word in dict1:
        word = word.decode('utf8')
        word = word.split()
        dict0[word[0]] = word[1]
    dict1.close()
    return dict0

dict0 = dict_read(DICT_PATH + 'dict1.txt', dict0)
dict0 = dict_read(DICT_PATH + 'dict2.txt', dict0) # correct words spelling dictionnary
slang = dict_read(DICT_PATH + 'slang_words.txt',slang) # slang words and their original forms

def remove_url_user_mention(text):
    # Remove URLs
    text = re.sub(r"<url>","",str(text))
    # Remove user's name
    text = re.sub(r"<user>","",str(text))
    # Remove mentions
    text = re.sub("@[^\s]*", "", str(text))
    return text.strip()

## Build positive and negative sentiment lexicon
#Source : https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
positiveWords = set(open(POS_NEG_WORDS_PATH + "positive-words.txt", encoding = "ISO-8859-1").read().split())
negativeWords = set(open(POS_NEG_WORDS_PATH + "negative-words.txt", encoding = "ISO-8859-1").read().split())

def emphasize_sentiment_words(text):
    '''
    This method adds the word positive/negative if the word processed is in the positive/negative dictionnary
    '''
    new_text = []
    for w in text.split():
        if w in positiveWords:
            new_text.append('positive ' + w)
        elif w in negativeWords:
            new_text.append('negative ' + w)
        else:
            new_text.append(w)
    return (" ".join(new_text)).strip()


def remove_one_letter_words(text):
    '''
    this method removes words with length less than two characters except for numbers
    '''
    return " ".join([w for w in text.split() if len(w) >1 or not w.isalpha()])

def remove_number(text):
    '''
    this method removes numbers from the tweets 
    '''
    text = re.sub("\d+", "", text)
    return text

def separate_hashtag(text):
    '''
    This method removes # from hashtags and if it's composed of more than one word 
    it will split it into different words using ninja library
    example: #nevergiveup ---> never give up
    '''
    separated = [] #ninja list
    text = re.sub(r"#", " #", text)
    new_text = []
    for w in text.split():
        if w.startswith("#"):
            w = re.sub(r'#(\S+)', r' \1 ', w)
            separated = wordninja.split(w)
            w = (" ".join(separated)).strip()
            new_text.append(w)
        else: new_text.append(w)
    return (" ".join(new_text)).strip()

def correct_words_from_dictionnary(text , dic):
    '''
    This method corrects words in the tweet using the slang/spelling dictionnaries aleready created
    '''
    text = text.split()
    for i in range(len(text)):
        if text[i] in dic.keys():
            text[i] = dic[text[i]]
    text = ' '.join(text)
    return text

punc_tokenizer = RegexpTokenizer(r'\w+') # tokenizer that removes punctuation 
def remove_punctuation(text):
    '''
    This method removes punctuation from the tweet
    '''
    return " ".join(punc_tokenizer.tokenize(text))


def remove_repetition(text):
    "This method removes letter repetitions from the tweet" 
    "tttthank youuu --> tthank youu"
    " then these words will be checked with spelling dictionnary "
    new_text=[]
    text=text.lower()
    text=text.split()
    for word in text:
        word = re.sub(r'(.)\1+', r'\1\1', word)
        new_text.append(word)
    text = ' '.join(new_text)    
    return text
        
def replace_haha(text):
    "hahaha --> positive "
    " this method will replace all forms of hahaha ( laughs) including those which have mistakes like hahajaha with the word positive"
    new_text=[]
    text=text.lower()
    text=text.split()
    haha= 'haha'
    pos = 'positive'
    for word in text:
        if haha in word :
            new_text.append(pos)
        else :
            new_text.append(word)
        
    text = ' '.join(new_text)    
    return text    

def emoji_translation(text):
    # Smile emojis= :), : ), :-), =) , (= , (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\=\)|\(\=|\(\s?:|\(-:|:\'\))', ' positive ', text)
    # Laugh emojis= :D, : D, :-D, xD, x-D, XD, X-D , xd
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D|xd)', ' positive ', text)
    # Love emojis= <3, :*
    text = re.sub(r'(<3|:\*)', ' positive ', text)
    # Wink emojis= ;-), ;), ;-D, ;D, (;,  (-; , :p
    text = re.sub(r'(;-?\)|;-?D|\(-?;|:p)', ' positive ', text)
    # Sad emojis= :-(, : (, :(, ):, ) : ,)-:
    text = re.sub(r'(:-\(|:\s?\(|\)\s?:|\)-:)', ' negative ', text)
    # Cry emojis= :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' negative ', text)
    return text

def apostrophe(text):
    '''
    This methode transforms words with apostrophes at the end into two words
    '''
    # Apostrophe lookup
    text = re.sub(r"i\'m", "i am ", text) # i'm --> i am
    text = re.sub(r"I\'m", "i am ", text) # I'm --> I am
    text = re.sub(r"it\'s","it is",str(text)) #it's --> it is
    text = re.sub(r"he\'s","he is",str(text)) #he's --> he is
    text = re.sub(r"she\'s","she is",str(text)) #she's --> she is
    text = re.sub(r"we\'re","we are",str(text)) #we're --> we are
    text = re.sub(r"they\'re","they are",str(text)) #they're --> they are
    
    text = re.sub(r"there\'s","there is",str(text)) #there's --> there is
    text = re.sub(r"that\'s","that is",str(text)) #that's --> that is
    
    text = re.sub(r"i\'d","i would",str(text)) #i'd --> i would
    text = re.sub(r"he\'d","he would",str(text)) #he'd --> he would
    text = re.sub(r"it\'d","it would",str(text)) #it'd --> it would
    text = re.sub(r"she\'d","she would",str(text)) #she'd --> she would
    text = re.sub(r"we\'d","we would",str(text)) #we'd --> we would
    text = re.sub(r"they\'d","they would",str(text)) #they'd --> they would
    
    text = re.sub(r"i\'ll","i will",str(text)) #i'll --> i will
    text = re.sub(r"he\'ll","he will",str(text)) #he'll --> he will
    text = re.sub(r"it\'ll","it will",str(text)) #it'll --> it will
    text = re.sub(r"she\'ll","she will",str(text)) #she'll --> she will
    text = re.sub(r"we\'ll","we will",str(text)) #we'll --> we will
    text = re.sub(r"they\'ll","they will",str(text)) #they'll --> they will
    
    text = re.sub(r"don\'t","do not",str(text)) #don't --> do not    
    text = re.sub(r"can\'t", "can not", text) #can't --> can not
    text = re.sub(r"cannot", "can not ", text) #cannot --> can not
    text = re.sub(r"could\'t", "could not", text) #could't --> could not
    text = re.sub(r"should\'t", "should not", text) #should't --> should not
    text = re.sub(r"haven\'t", "have not", text) #haven't --> have not
    text = re.sub(r"didn\'t", "did not", text) #didn't --> did not    
    
    text = re.sub(r"what\'s", "what is", text) #what's --> what is
    text = re.sub(r"where\'s", "where is", text) #where's --> where is
    text = re.sub(r"when\'s", "when is", text) #when's --> when is
    text = re.sub(r"why\'s", "why is", text) #why's --> why is   
    
    
    
    return text

# Loading stopwords list from NLTK
stoplist = set(stopwords.words("english"))
negative_stopwords = ['no', 'not', 'nor', 'only', 'against', 'up', 'down', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ain', 'aren', 'mightn', 'mustn', 'needn', 'shouldn', 'wasn', 'weren', 'wouldn']
## Remove negation stopwords ( they have a negative sentiment )
for w in negative_stopwords:
    stoplist.remove(w)

def remove_stopwords(text):
    '''
    This method removes stop words from the tweet
    '''
    new_text = text.split()
    for word in new_text:
        if word in stoplist:
            new_text.remove(word)
    return ' '.join(new_text)

#  Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatizing(text):
    '''
    lemmatize words: dances, dancing ,danced --> dance
    '''
    words = text.split()
    lemmatized = list()
    for word in words:
        try:
            lemmatized.append(lemmatizer.lemmatize(word).lower())  #check problem doesnt work correctly
        except Exception:
             lemmatized.append(word)
    return " ".join(lemmatized)

# Stemming
stemmer = PorterStemmer()
def stemming(text):
    '''
    stemm words: Car, cars --> car
    '''
    x = [stemmer.stem(t) for t in text.split()]
    return " ".join(x)

def load_cleaned_data_and_test(positive_data_file, negative_data_file, test_data_file, HASHTAG = True, EMPHASIZE = True, PUNC=True, NUMBER =True, SMALL_WORDS = True , \
                        SLANG =True, APOSTROPHE = True, EMOJI = True, REPITITION = True, SPELLING = True, \
                        STOPWORDS = True, LEMMATIZE = True, STEMMING = True):
    
    # Load data from files
    positive_tweets = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_tweets = [s.strip() for s in positive_tweets]
    negative_tweets = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_tweets = [s.strip() for s in negative_tweets]
    test_data = list(open(test_data_file, "r", encoding='utf-8').readlines())
    test_data = [s.strip() for s in test_data]
    
    # Split by words
    print("Starting Data processing")
    x = positive_tweets + negative_tweets
    
    # remove urls , user name and mentions
    print("Processing urls , user names and mentions")
    positive_tweets = [remove_url_user_mention(sentence) for sentence in positive_tweets]
    negative_tweets = [remove_url_user_mention(sentence) for sentence in negative_tweets]
    test_data = [remove_url_user_mention(sentence) for sentence in test_data]
    
    if EMOJI:
        print("Processing emojis")
        positive_tweets = [emoji_translation(sentence) for sentence in positive_tweets]
        negative_tweets = [emoji_translation(sentence) for sentence in negative_tweets]
        test_data = [emoji_translation(sentence) for sentence in test_data]
        
        
    if REPITITION:
        print("Processing repetition")
        positive_tweets = [remove_repetition(sentence) for sentence in positive_tweets]
        negative_tweets = [remove_repetition(sentence) for sentence in negative_tweets]
        test_data = [remove_repetition(sentence) for sentence in test_data]  
        print("Processing haha..s")
        positive_tweets = [replace_haha(sentence) for sentence in positive_tweets]
        negative_tweets = [replace_haha(sentence) for sentence in negative_tweets]
        test_data = [replace_haha(sentence) for sentence in test_data]
        
    if SLANG:
        print("Processing slang words")
        positive_tweets = [correct_words_from_dictionnary(sentence, slang ) for sentence in positive_tweets]
        negative_tweets = [correct_words_from_dictionnary(sentence, slang ) for sentence in negative_tweets]
        test_data = [correct_words_from_dictionnary(sentence,slang ) for sentence in test_data]
        
    if SPELLING:
        print("Processing spelling mistakes")
        positive_tweets = [correct_words_from_dictionnary(sentence, dict0 ) for sentence in positive_tweets]
        negative_tweets = [correct_words_from_dictionnary(sentence, dict0 ) for sentence in negative_tweets]
        test_data = [correct_words_from_dictionnary(sentence, dict0 ) for sentence in test_data]
        
       
        
    if HASHTAG:
        print("processing hashtags")
        positive_tweets = [separate_hashtag(sentence) for sentence in positive_tweets]
        negative_tweets = [separate_hashtag(sentence) for sentence in negative_tweets]
        test_data = [separate_hashtag(sentence) for sentence in test_data]  
        
    if STOPWORDS:
        print("Processing stop words")
        positive_tweets = [remove_stopwords(sentence) for sentence in positive_tweets]
        negative_tweets = [remove_stopwords(sentence) for sentence in negative_tweets]
        test_data = [remove_stopwords(sentence) for sentence in test_data]    
        
    if APOSTROPHE:
        print("Processing apostrophes")
        positive_tweets = [apostrophe(sentence) for sentence in positive_tweets]
        negative_tweets = [apostrophe(sentence) for sentence in negative_tweets]
        test_data = [apostrophe(sentence) for sentence in test_data]   
    
    if EMPHASIZE:
        print("Processing emphasize sentiment words")
        positive_tweets = [emphasize_sentiment_words(sentence) for sentence in positive_tweets]
        negative_tweets = [emphasize_sentiment_words(sentence) for sentence in negative_tweets]
        test_data = [emphasize_sentiment_words(sentence) for sentence in test_data]

    if PUNC:
        print("Processing punctuation")
        positive_tweets = [remove_punctuation(sentence) for sentence in positive_tweets]
        negative_tweets = [remove_punctuation(sentence) for sentence in negative_tweets]
        test_data = [remove_punctuation(sentence) for sentence in test_data]
    
    if NUMBER:
        print("Processing numbers")
        positive_tweets = [remove_number(sentence) for sentence in positive_tweets]
        negative_tweets = [remove_number(sentence) for sentence in negative_tweets]
        test_data = [remove_number(sentence) for sentence in test_data]
    
    if SMALL_WORDS:
        print("Processing small words")
        positive_tweets = [remove_one_letter_words(sentence) for sentence in positive_tweets]
        negative_tweets = [remove_one_letter_words(sentence) for sentence in negative_tweets]
        test_data = [remove_one_letter_words(sentence) for sentence in test_data]   
    

    if SPELLING:
        print("Processing spelling mistakes a second time")
        positive_tweets = [correct_words_from_dictionnary(sentence, dict0 ) for sentence in positive_tweets]
        negative_tweets = [correct_words_from_dictionnary(sentence, dict0 ) for sentence in negative_tweets]
        test_data = [correct_words_from_dictionnary(sentence, dict0 ) for sentence in test_data]
    
    if LEMMATIZE:
        print("Processing Lemmatizing")
        positive_tweets = [lemmatizing(sentence) for sentence in positive_tweets]
        negative_tweets = [lemmatizing(sentence) for sentence in negative_tweets]
        test_data = [lemmatizing(sentence) for sentence in test_data]
        
    if STEMMING:
        print("Processing Stemming")
        positive_tweets = [stemming(sentence) for sentence in positive_tweets]
        negative_tweets = [stemming(sentence) for sentence in negative_tweets]
        test_data = [stemming(sentence) for sentence in test_data]
    print("Returning Positive X data , Negative X data and test data")
    return [positive_tweets, negative_tweets, test_data]

    
        
    
        
    
        
    
        
    
        
