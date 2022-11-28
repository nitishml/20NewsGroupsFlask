#Imports
from sklearn.datasets import fetch_20newsgroups
import pickle

#prequesits
cats = ['rec.autos','sci.space','sci.electronics']
train = fetch_20newsgroups(subset="train", categories= cats)

model = pickle.load(open('model.pkl','rb'))

#funtion to generate labels
def predicter(s, train=train, model=model):
	pred = model.predict([s])
	return train.target_names[pred[0]]

print("Enter text to predict:")
inp = input()
print("This text would have been said in the channel:")
print(predicter(inp))