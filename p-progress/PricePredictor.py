# this class collects features and predicts the price in the next dt time.
# implement algorithms including bayesian regression]
from sklearn import linear_model
from sklearn.cross_validation import KFold

non_price_inputs =  ['avg-confirmation-time.txt', 'estimated-transaction-volume.txt', 'my-wallet-transaction-volume.txt', 'total-bitcoins.txt', 
             'bitcoin-days-destroyed-cumulative.txt','hash-rate.txt', 'n-orphaned-blocks.txt','trade-volume.txt', 'bitcoin-days-destroyed.txt','market-cap.txt', 
             'n-transactions-excluding-popular.txt','transaction-fees.txt', 'blocks-size.txt','n-transactions-per-block.txt', 'tx-trade-ratio.txt', 
             'cost-per-transaction.txt','miners-revenue.txt', 'n-transactions.txt', 'difficulty.txt','my-wallet-n-tx.txt', 'n-unique-addresses.txt', 
             'estimated-transaction-volume-usd.txt', 'my-wallet-n-users.txt', 'output-volume.txt']

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

