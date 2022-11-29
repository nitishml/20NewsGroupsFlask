#Imports
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
class NewsNlp:
	selected_categories = []
	train = []
	#function to set category names
	def set_categories(self,categories):
		self.selected_categories = categories

	#function to get category names
	def get_categories(self):
		return self.selected_categories

	#function to train
	def training(self,cats):
		self.train = fetch_20newsgroups(subset="train", categories= cats)
		model = make_pipeline(TfidfVectorizer(),MultinomialNB())
		model.fit(self.train.data, self.train.target)
		self.set_categories(cats)
		pickle.dump(model, open('model.pkl','wb'))

	#function to predict
	def predict(self,s):
		model = pickle.load(open('model.pkl','rb'))
		pred = model.predict([s])
		cats = self.get_categories()
		return self.train.target_names[pred[0]]


