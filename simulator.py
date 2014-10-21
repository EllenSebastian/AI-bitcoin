from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
import numpy as np

def SGD_example():
	X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
	Y = np.array([1, 1, 2, 2])
	clf = linear_model.SGDClassifier()
	clf.fit(X, Y)
	print(clf.predict([[-0.8, -1]]))


# return an array of features 
def get_features(timestamp):


def get_bitcoin_price_at(timestamp): 


# predict the value of bitcoin at timestamp_future, using data from timestamp_now
def predict(timestamp_now, timestamp_future): 



def simulate(timestamp_now, timestamp_future): 
	prediction = predict(timestamp_now, timestamp_future)
	actual_price = get_bitcoin_price_at(timestamp_future)