import numpy as np
import pandas as pd
import re
import string
import nltk
import joblib as jb
import streamlit as stlit
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("tweets.csv")

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate Speech or Offensive Language"})

data = data[["tweet", "labels"]]

stemmer = nltk.SnowballStemmer("english")

stopword=set(stopwords.words('english'))

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

data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = DecisionTreeClassifier()
model.fit(X_train,y_train)

jb.dump(model,'model.h5')


def classify(text):
    model=jb.load('model.h5')
    classification = model.predict(text)
    stlit.title(str(classification[0]))

def hate_speech_detection_webapp():

    stlit.title("Hate Speech Detection")
    text = stlit.text_area("Enter the Text/Tweet/Comment : ")
    if stlit.button("Classify"):
        text = cv.transform([text]).toarray()
        classify(text)
        
        
hate_speech_detection_webapp() 