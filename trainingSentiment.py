# Author: Eliska Kloberdanz
# import libraries

from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import urljoin
import requests
import re
from datetime import date, timedelta
import pandas as pd
import pprint
import pickle
import datetime
import nltk


def log(message):
    date_time = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    print('{date_time} {message}'.format(date_time=date_time, message=message))

log('starting running')

import nltk
import os
import re
import pprint
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Step 1 – Training data

base_directory = '/home/eliska/Iowa State/Principles of AI/Stock-Picker/MovieTrain/train'
sentences = []
pos_sentences = []
neg_sentences = []
for directory in ['pos', 'neg']:
    full_directory = os.path.join(base_directory, directory)
    for filename in os.listdir(full_directory):
        filename = os.path.join(full_directory, filename)
        with open(filename, 'r') as f:
            paragraph = f.read()
            for sentence in ((sentence.strip(), directory)
                             for sentence in re.split('\.|!|\?', paragraph) if sentence):
                if directory == 'pos':
                    pos_sentences.append(sentence)
                else:
                    neg_sentences.append(sentence)
                #sentences.append(sentence)

#train = sentences
train = pos_sentences[:1000] + neg_sentences[:1000]

# Step 2 
dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))


# Step 3
t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
  
# Step 4 – the classifier is trained with sample data
classifier = nltk.NaiveBayesClassifier.train(t)

log('training finished')

save_classifier = open("naivebayes1.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

log('model saved')



