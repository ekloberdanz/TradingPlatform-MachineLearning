# Import libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os
from sklearn.model_selection import GridSearchCV
import pickle
import sys

# Import data
SandP_train = sys.argv[1]
dataset = pd.read_csv(SandP_train)
dataset = dataset.dropna()
dataset = dataset[['Open', 'High', 'Low', 'Close']]

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


#df['Rate of Change'] = (df['Close'] - df['Close'].['Close']shift(10))/df['Close']
dataset['Price_Rise'] = np.where(dataset['Close'].shift() < dataset['Close'], 1, 0)
#dataset.to_csv('testoutput.csv')
dataset = dataset.dropna()


# Dataset storing input variables
X = dataset.iloc[:, :-1]
#X = dataset.iloc[:, dataset.columns != 'Date']
#X = X.iloc[:, dataset.columns != 'Price_Rise']

# Dataset storing output variable Price_Rise
y = dataset.iloc[:, -1]

# Split data into training (80%) and testing set (20%)
# split = int(len(dataset)*0.8)
# X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Build NN
#best fit so far:
#mlp = MLPClassifier(alpha=0.01, random_state=1, activation='logistic',learning_rate_init=0.0005, solver='adam', hidden_layer_sizes=[20, 15, 10, 5], max_iter=300, early_stopping=True, validation_fraction=0.1, batch_size=200)
mlp = MLPClassifier(alpha=0.001, random_state=1, activation='logistic',learning_rate_init=0.001, solver='adam', hidden_layer_sizes=[20, 15, 10, 5], max_iter=300, early_stopping=True, validation_fraction=0.1, batch_size=200, verbose=True)
mlp.fit(X_train_scaled, y_train)

# Save model as a pickle
save_mlp = open("NN.pickle","wb")
pickle.dump(mlp, save_mlp)
save_mlp.close()

predictions_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions_test))
print(classification_report(y_test, predictions_test))
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))
print("Accuracy on entire data set: {:.3f}".format(mlp.score(X, y)))


# Compare strategy and market returns
y_pred = mlp.predict(X)
dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
trade_dataset = dataset.dropna()

trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

trade_dataset['Strategy Returns'] = 0.
#trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == trade_dataset['Price_Rise'], trade_dataset['Tomorrows Returns'], -trade_dataset['Tomorrows Returns'])
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == trade_dataset['Price_Rise'], trade_dataset['Tomorrows Returns'], 0)

trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()


# See how the learning function changes over time for training set
# loss_values = mlp.loss_curve_
# plt.plot(loss_values)
# mlp.fit(X_train,y_train)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.001))
plt.plot(mlp.loss_curve_)
# mlp.fit(X_test,y_test)
# plt.plot(mlp.loss_curve_)
plt.show()

trade_dataset.to_csv('TechAnalysis.csv')

#Links
#https://stats.stackexchange.com/questions/255105/why-is-the-validation-accuracy-fluctuating
#https://stackoverflow.com/questions/12182063/how-to-calculate-the-regularization-parameter-in-linear-regression
#https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
