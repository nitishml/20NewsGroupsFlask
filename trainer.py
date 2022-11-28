#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pickle

#Dataset taken from sklearn cloud
df = fetch_20newsgroups()
print("Available news channels in database : ")
print(df.target_names)

selected_channels = []
cats = ['rec.autos','sci.space','sci.electronics']
print("Selected categories: ")
print(cats)
print("Training the model on these categories")

train = fetch_20newsgroups(subset="train", categories= cats)
test = fetch_20newsgroups(subset="test", categories= cats)

model = make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(train.data, train.target)
print("Model trained")

labels = model.predict(test.data)
print("The accuracy of the trained model is:")
print(metrics.accuracy_score(test.target, labels))

pickle.dump(model, open('model.pkl','wb'))

