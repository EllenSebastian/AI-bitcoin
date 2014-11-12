# this class collects features and predicts the price in the next dt time.
# implement algorithms including bayesian regression]
from sklearn import linear_model
from sklearn.cross_validation import KFold

class PricePredictor: 
	# train the algorithm to predict the next deltaP over the next numSeconds seconds. 
	# trainSet will include: 
	#   [{features: [val1, val2, val3], y: Y1 }],...  where the Y values happen in nSeconds seconds 
	# algorithm can be "linear"
	# foreach set of features, need the price that happens in X seconds. --> Y is the price in X seconds instead of hte current price. 
	def __init__(self, trainX, trainY, algorithm, nSeconds=60, featureLabels=[]):
		self.algorithm = algorithm.lower()
		self.nSeconds = nSeconds 
		self.featureLabels = featureLabels
		self.trainX = trainX
		self.trainY = trainY
		if self.algorithm == 'linear': 
			self.model = linear_model.LinearRegression()
			self.model.fit (trainX, trainY)
	def predict(self, features): 
		if self.aglorithm == 'linear': 
		    reg = linear_model.LinearRegression()
    		return reg.predict(features) # value predicted in dt seconds. 
	def crossValidation(self, n): 
		kf = KFold(len(self.trainX), n_folds = n)
		total_error = 0
		predictions = {}
		for train,test in kf: 
		    print test
		    this_x = []
		    this_y = []
		    for i in train: 
		        this_x.append(self.trainX[i])
		        this_y.append(self.trainY[i])
		    reg = linear_model.LinearRegression()
		    reg.fit(this_x, this_y)
		    for test_i in test: 
			    predicted = reg.predict(self.trainX[test_i])
			    predictions[test_i] = predicted
			    squared_error = (predicted - self.trainY[test_i])**2
		    total_error += squared_error
		return total_error / len(self.trainX), predictions

err, predictions= pp_linear.crossValidation(10)
pp_linear = PricePredictor(all_features, all_Y, 'linear')
false_neg, true_neg, false_pos, true_pos = 0,0,0,0
for i in range(0,9999): 
	if all_Y[i] < 0 and predictions[i] < 0: 
		true_neg += 1
	elif all_Y[i] > 0 and predictions[i] > 0: 
		true_pos += 1
	elif predictions[i] > 0: 
		false_pos += 1 
		print predictions[i], all_Y[i]
	elif predictions[i] < 0: 
		false_neg += 1 
		print predictions[i], all_Y[i]
	else: 
		print '???????', predictions[i], all_Y[i]
