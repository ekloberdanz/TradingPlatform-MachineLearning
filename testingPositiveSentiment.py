import pickle
import os
import nltk
import pandas as pd
import numpy as np
import sys

movieDirectory = sys.argv[1]
listOfRatings = []

NLPmodel = sys.argv[2]
f = open(NLPmodel, 'rb')
classifier = pickle.load(f)

for filename in os.listdir(movieDirectory):
    filename = os.path.join(movieDirectory, filename)
    sumOfPositive = 0
    sumOfNegative = 0
    
    with open(filename, 'r') as f:
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

Xtest = listOfRatings
ytest = []
r = len(listOfRatings)
for i in range (r):
    ytest.append('pos')

data = {'Predicted Sentiment':Xtest, 'Actual Sentiment': ytest}
dataFrame = pd.DataFrame(data)
print(dataFrame)
dataFrame['Compare'] = np.where(dataFrame['Predicted Sentiment'] == dataFrame['Actual Sentiment'], "true", "false")

totalNumberOfDataPoints = dataFrame.shape[0]

succesfulComparison = dataFrame.loc[dataFrame['Compare'] == 'true']
successfulNumberOfDataPoints = succesfulComparison.shape[0]


accuracy = successfulNumberOfDataPoints/totalNumberOfDataPoints
print('Model accuracy on a positive test dataset is', accuracy)


'''
X_test = dataFrame.iloc[:, :-1]

y_test = dataFrame.iloc[:, dataFrame.columns != 'Predicted Sentiment']
   
comparison = []
#line = 1
for line in dataFrame:
    if dataFrame.iloc[1, 0] == dataFrame.iloc[1, 1]:
        comparison.append('true')
    else:
        comparison.append('false')
    #line += 1
print(comparison) 

'''
