Developed a neural net for technical analysis and a natural language processing model for sentiment analysis of webscraped news articles. Achieved returns 4% above market.

All programs must be run with Python3. I used Python 3.5.2 on Ubuntu 16.04.

The code base contains two machine learning models: 
(1) Neural Net for predicting if S&P 500 index will rise or fall
(2) Natural Language Processing (NLP) for assessign if market sentiment is positive or negative

These two models are then combined to generate two trading strategies: Buy & Sell, and Buy & Hold. 


List of programs:

(1) NN.py: Neural Net that loads S&P_train.csv as training data and outputs a trained model NN.pickle. Takes path to S&P_train.csv as argument form command line

(2) trainingSentiment.py: Natural Language Processing that loads posive and negative movie reviews from a directory and outputs a trained NLP model naivebayes1.pickle
Takes a long time to run on a powerful machine.

(3) testingPositiveSentiment.py: Tests the accuracy of the NLP model on 12,000 positve movie reviews. Takes two arguments from command line: the directory where positive movie reviews are and naivebayes1.pickle 
Example: python3 testingNegativeSentiment.py '/home/eliska/Iowa State/Principles of AI/Stock-Picker/MovieTrain/test/pos' 'naivebayes1.pickle'

(4) testingNegativeSentiment.py: Tests the accuracy of the NLP model on 12,000 negative movie reviews. Takes two arguments from command line: the directory where negative movie reviews are and naivebayes1.pickle 
Example: python3 testingNegativeSentiment.py '/home/eliska/Iowa State/Principles of AI/Stock-Picker/MovieTrain/test/neg' 'naivebayes1.pickle'

(5) downloadAndSaveArticles.py: Webscrapes five years worth of daily Wall Street Journal articles and saves them in a directory. Takes one argument from command line: the directory where you want the articles to be saved.


(6) htmlClassify.py: Applies the NLP model (naivebayes1.pickle) to the downloaded articles, classifies each day as posivie or negative, and outputs a csv file wsjSentiment.csv
Can be run from terminal with two arguments: the file path to Articles and naivebayes1.pickle. Example: python3 htmlClassify.py '/home/eliska/Iowa State/Principles of AI/Stock-Picker/Articles' 'naivebayes1.pickle'

(7) TechAndSent.py: Final program that combines the Neural Net and NLP models. It loads S&P_test.csv as input for the Neural Net that is called from the NN.pickle file. It also loads wsjSentiment.csv, which are the results from the NLP model. Then it simulates two trading strategies: Buy&Sell and Buy&Hold. 
In case of the first strategy, a decision to buy is made when either the neural net predicts that the next day price will increase or when the natural language processing model assesses that the market sentiment is positive. If neither condition applies, the model makes a decision to sell. The second strategy works the same way, except for it just holds a position instead of selling when neither condition applies. 


Instructions for TA on how to run the program:
Install dependencies: pip install -r requirements.txt 

Please run TechAndSent.py from the terminal with three arguments as follows: python3 TechAndSent.py "NN.pickle" "S&P_test.csv" "wsjSentiment.csv". 
This script uses the results of the six other programs (trained NN model, trained NLP model, classified Wall Street Journal articles) as input and produces the final product of this project.
