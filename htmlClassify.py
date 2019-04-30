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
import os
import sys


articlesDirectory = sys.argv[1]
listOfRatings = []

NLPmodel = sys.argv[2]
f = open(NLPmodel, 'rb')
classifier = pickle.load(f)

ListOfArticles = []

for filename in os.listdir(articlesDirectory):
    filename = os.path.join(articlesDirectory, filename)
    sumOfPositive = 0
    sumOfNegative = 0
    
    with open(filename, 'r') as f:
        ListOfArticles.append(f)
        for line in f:
            feats = dict([(word, True) for word in line])
      
            result = classifier.classify(feats)
                
                
        if result == "pos":
            sumOfPositive += 1
        else:
            sumOfNegative += 1
    if sumOfPositive > sumOfNegative:
        listOfRatings.append("pos")
    else:
        listOfRatings.append("neg")


#print(listOfRatings)
f.close()

startDate = date(2013, 10, 24)
endDate = date (2018, 10, 26)

numberOfDays = endDate - startDate
listOfDays = []
for i in range(numberOfDays.days + 1):
    listOfDays.append(startDate + timedelta(i))

weekdays = [0, 1, 2, 3, 4]
listOfWeekDays = []
for date in listOfDays:
    if date.weekday() in weekdays:
        listOfWeekDays.append(date)


        
data = {'Date':listOfWeekDays, 'Article': ListOfArticles, 'Sentiment': listOfRatings}

dataFrame = pd.DataFrame(data)

dataFrame.to_csv('wsjSentiment.csv')



