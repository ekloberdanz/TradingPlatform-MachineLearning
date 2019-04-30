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


site = 'http://www.wsj.com/public/page/archive-'

ListOfArticles = []
    
d = argv[1] # directory where to save articles
for day in listOfWeekDays:
    f = open(d+str(day)+'.txt', 'w')
    day = str(day)
    r = requests.get(site+day+'.html')
    headlines = r.text
    souplink = BeautifulSoup(headlines, 'lxml')
    tags = souplink.findAll('p')
    DayText = []
    for t in tags:
        DayText.append(t.text)
        f.write(t.text)
    ListOfArticles.append(DayText)
    f.close()
    
#print(ListOfArticles)    
    
print('articles are downloaded')
