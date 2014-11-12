# this class collects features and predicts the price in the next dt time.
# implement algorithms including bayesian regression]

class PricePredictor: 

	# train the algorithm to predict the next deltaP over the next numSeconds seconds. 
	# trainSet will include: 
	#   [{features: [val1, val2, val3], y: Y1 }],...  where the Y values happen in nSeconds seconds 
	# algorithm can be "linear"
	# foreach set of features, need the price that happens in X seconds. --> Y is the price in X seconds instead of hte current price. 
	def __init__(trainSet, algorithm, nSeconds, featureLabels=[]):
		self.aglorithm = algorithm.lower()
		self.nSeconds = nSeconds 
		self.featureLabels = featureLabels
		if self.algorithm == 'linear': 
		    self.model = linear_model.LinearRegression()
			self.model.fit ([x['features'] for x in trainSet], [y['y'] for y in trainSet])

	def predict(self, features): 
		if self.aglorithm == 'linear': 
		    reg = linear_model.LinearRegression()
    		reg.fit(this_x, this_y)
    		return reg.predict(all_features_clean[test[0]]) # value predicted in dt seconds. 


"The learning of w is done simply by finding the best linear fit over all choices given the selection of Sj , 1 ≤ j ≤ 3"