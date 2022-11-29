#imports
import numpy as np
from flask import Flask, render_template,request
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from NewsNlp import NewsNlp

#globals
app = Flask(__name__)

#our created class
news = NewsNlp()

#index
@app.route('/')
def home():
    return render_template('index.html')

#training
@app.route('/train', methods=["POST"])
def train():
    cats = request.form.getlist('check')
    news.training(cats)
    message="successful"
    return render_template('index.html', message=message)

#predicting
@app.route('/predict',methods=['POST'])
def predict():
    input_string = request.form["textToPredict"]
    predicted = news.predict(input_string)
    return render_template('index.html', prediction_text=predicted)

if __name__ == "__main__":
    app.run(debug=True)