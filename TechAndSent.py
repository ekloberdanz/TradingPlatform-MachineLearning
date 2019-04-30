# Import libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os
from sklearn.model_selection import GridSearchCV
import pickle
import datetime
import sys


NNmodel = sys.argv[1]
f = open(NNmodel, 'rb')
mlp = pickle.load(f)

SandP_test = sys.argv[2]
priceData = pd.read_csv(SandP_test)

dataset = priceData[['Date', 'Open', 'High', 'Low', 'Close']]
dataset = dataset.dropna()

# Add financial indicators to dataset
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day SMA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day SMA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['20day SMA'] = dataset['Close'].shift(1).rolling(window = 20).mean()
# dataset['30day SMA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev'] = dataset['Close'].rolling(5).std()
dataset['5day EMA'] = dataset['Close'].ewm(span=5, adjust=False).mean() #short EMA
dataset['50day EMA'] = dataset['Close'].ewm(span=50, adjust=False).mean() #long EMA
# dataset['20day EMA'] = dataset['Close'].ewm(span=20, adjust=False).mean() #long EMA
dataset['Bollinger Percent'] = (dataset['Close'] - (dataset['20day SMA']-(2*dataset['Close'].rolling(20).std())))/((dataset['20day SMA']+(2*dataset['Close'].rolling(20).std())) - (dataset['20day SMA']-(2*dataset['Close'].rolling(20).std())))
dataset['Momentum'] = dataset['Close'] - dataset['Close'].shift(-10)
dataset = dataset.dropna()

modelData = dataset.iloc[:, dataset.columns != 'Date']
# Pass data into model
pricePred = mlp.predict(modelData)

dataset['Price Prediction'] = np.NaN
dataset.iloc[(len(dataset) - len(pricePred)):,-1:] = pricePred


# Import data
WSJ = sys.argv[3]
sentData = pd.read_csv(WSJ)
sentData = sentData.dropna()


joinedData = pd.merge(dataset, sentData, how='outer', left_on = "Date", right_on = "Date")
#print(joinedData)
joinedData.to_csv('JoinedData.csv')
f.close()

trade_dataset = joinedData.dropna()

trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

# Assumes buy and sell
trade_dataset['BuySell Strategy Returns'] = 0.
trade_dataset['BuySell Strategy Returns'] = np.where((trade_dataset['Price Prediction'] == 1) | (trade_dataset['Sentiment'] == "pos"), trade_dataset['Tomorrows Returns'], -trade_dataset['Tomorrows Returns'])

#Assumes buy and hold
trade_dataset['BuyHold Strategy Returns'] = 0.
trade_dataset['BuyHold Strategy Returns'] = np.where((trade_dataset['Price Prediction'] == 1) |(trade_dataset['Sentiment'] == "pos"), trade_dataset['Tomorrows Returns'], 0)

trade_dataset['Cumulative Market Returns'] = (np.cumsum(trade_dataset['Tomorrows Returns']))*100
trade_dataset['Cumulative BuySell Strategy Returns'] = (np.cumsum(trade_dataset['BuySell Strategy Returns']))*100
trade_dataset['Cumulative BuyHold Strategy Returns'] = (np.cumsum(trade_dataset['BuyHold Strategy Returns']))*100


import matplotlib.pyplot as plt
xAxis = [datetime.datetime.strptime(day, '%Y-%m-%d') for day in trade_dataset['Date']]


plt.figure(figsize=(10,5))
plt.plot(xAxis, trade_dataset['Cumulative Market Returns'],color='r', label='Market Returns' )
plt.plot(xAxis, trade_dataset['Cumulative BuySell Strategy Returns'],color='g', label='BuySell Strategy Returns')
plt.plot(xAxis, trade_dataset['Cumulative BuyHold Strategy Returns'],color='b', label='BuyHold Strategy Returns')
#plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
#plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.xlabel('Time')
plt.ylabel('Return in %')
plt.title('Buy and Sell Strategy')
plt.legend()
plt.show()

#trade_dataset.to_csv('final.csv')
averageMarketReturn = (np.mean(trade_dataset['Cumulative Market Returns']))
print('The average cummulative market return between 10/24/2013 and 10/26/2018 is',round(averageMarketReturn, 2), '%')
averageBuySellStrategyReturn = (np.mean(trade_dataset['Cumulative BuySell Strategy Returns']))
print('The average cummulative BuySell strategy return between 10/24/2013 and 10/26/2018 is',round(averageBuySellStrategyReturn, 2), '%')

averageBuyHoldStrategyReturn = (np.mean(trade_dataset['Cumulative BuyHold Strategy Returns']))
print('The average cummulative BuyHold strategy return between 10/24/2013 and 10/26/2018 is',round(averageBuyHoldStrategyReturn, 2), '%')

