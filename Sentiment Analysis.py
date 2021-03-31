import pandas as pd
import numpy as np
import datetime as dt
import re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from datetime import datetime

sia = SentimentIntensityAnalyzer()
def getSIA(text):
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    polarity = blob.sentiment.polarity
    sentiment = sia.polarity_scores(text)
    
    compound = sentiment['compound']
    pos = sentiment['pos']
    neu = sentiment['neu']
    neg = sentiment['neg']

    return [subjectivity,polarity,compound,pos,neu,neg]

#Clean le data
def clean_headlines(headlines):
    headlines = re.sub('b[(\')]','',headlines)
    headlines = re.sub('b[(\")]','',headlines)
    headlines = re.sub("\'",'',headlines)
    return headlines

#Load data 
news = pd.read_csv('News_DJIA.csv')
price = pd.read_csv('Value_DJIA.csv')

#Combiner les 25 colonnes de headlines en une colonne 
title_cols = list(news.columns[2:24]) 
news['News'] = news[title_cols].agg(' '.join, axis = 1)

#Clean le data 
news['News'] = news.apply(lambda x: clean_headlines(x['News']), axis = 1)

#Retirer les 25 colonnes 
news = news.drop(news.columns[2:27], axis = 1)

#Add sentiment
news[['subjectivity','polarity','compound','pos','neu','neg']] = news.apply(lambda x: pd.Series(getSIA(x['News'])),axis = 1)
    
# merge les dataframe des prix et headlines
data = news.merge(price, on='Date')

# Date to datetime object
data['Date'] = pd.to_datetime(data['Date'])

# split news to train and test
training_size = (0.8 * data.shape[0])
train = data[data.index <= training_size]
test = data[data.index > training_size]
x_cols = ['Open','Close','High','Low','subjectivity','polarity','compound','pos','neu','neg']
X_train, y_train = np.array(train[x_cols]), np.array(train['Label'])
X_test, y_test = np.array(test[x_cols]), np.array(test['Label'])

# normaliser data
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

#Différents modèles
models = {  'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
            'SVM Classification': SVC(),
            'SGDClassifier': SGDClassifier(loss="hinge", penalty="l2", max_iter=100),
            'KNeighborsClassifier':KNeighborsClassifier(n_neighbors=10),
            'GaussianProcessClassifier': GaussianProcessClassifier(),
            }

for model_name in models.keys():
    
    model = models[model_name]
    print('--------------',model_name,'---------------')
    model.fit(X_train,y_train)
    print(classification_report(model.predict(X_test),y_test))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    