import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk import FreqDist, bigrams
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
import matplotlib.pyplot as plt

data = pd.read_excel('C:/Users/User/Desktop/Online Retail Data Set.xlsx')
print(data.head())
print(data.columns)
data = data[["Description"]]
data.isnull().sum()

# Download stopwords
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

# Cleaning dataset
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["Description"] = data["Description"].apply(clean) 

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words):
    cleaned_tokens = []
    for token in tweet_tokens:
        token = re.sub('http[s]','', token)
        token = re.sub('//t.co/[A-Za-z0-9]+','', token)
        token = re.sub('(@[A-Za-z0-9_]+)','', token)
        token = re.sub('[0-9]','', token)
        if (len(token) > 3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

stop_words = stopwords.words('english')

data["Description"] = data["Description"].str.replace('(@[A-Za-z0-9_]+)', '', regex=True)
data_token = data["Description"].apply(word_tokenize).tolist()

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

cleaned_tokens = []
for tokens in data_token:
    rm_noise=remove_noise(tokens, stop_words)
    lemma_tokens=lemmatize_sentence(rm_noise)
    cleaned_tokens.append(lemma_tokens)
    
def get_all_words(cleaned_tokens_list):
    tokens=[]
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token
        
tokens_flat=get_all_words(cleaned_tokens)