# this class collects features and predicts the price in the next dt time.
# implement algorithms including bayesian regression]
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.gaussian_process import GaussianProcess
import numpy as np

non_price_inputs =  ['avg-confirmation-time.txt', 'estimated-transaction-volume.txt', 'my-wallet-transaction-volume.txt', 'total-bitcoins.txt', 
                     'bitcoin-days-destroyed-cumulative.txt','hash-rate.txt', 'n-orphaned-blocks.txt','trade-volume.txt', 'bitcoin-days-destroyed.txt','market-cap.txt', 
                     'n-transactions-excluding-popular.txt','transaction-fees.txt', 'blocks-size.txt','n-transactions-per-block.txt', 'tx-trade-ratio.txt', 
                     'cost-per-transaction.txt','miners-revenue.txt', 'n-transactions.txt', 'difficulty.txt','my-wallet-n-tx.txt', 'n-unique-addresses.txt', 
                     'estimated-transaction-volume-usd.txt', 'my-wallet-n-users.txt', 'output-volume.txt']

class PricePredictor: 
    # train the algorithm to predict the next deltaP over the next numSeconds seconds. 
    def __init__(self, trainX, trainY, algorithm, nSeconds=60, featureLabels=[]):
        self.algorithm = algorithm.lower()
        self.nSeconds = nSeconds 
        self.featureLabels = featureLabels
        self.trainX = trainX
        self.trainY = trainY
        self.model = self.model_for_algorithm()
        #pdb.set_trace()
        self.model.fit (trainX, trainY)


    def model_for_algorithm(self): 
        if self.algorithm == 'linear': 
            return linear_model.LinearRegression()
        elif self.algorithm == 'gp':
            return GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        elif self.algorithm == 'bayesianridge': 
            return linear_model.BayesianRidge()
        elif self.algorithm == 'ridge': 
            return linear_model.Ridge (alpha = 1)
        elif self.algorithm == 'logistic':
            return linear_model.LogisticRegression()
        elif self.algorithm == 'perceptron':
            return linear_model.Perceptron()


    def train(self):
        self.model.fit(self.trainX, self.trainY)


    def predict(self, features): 
        if self.algorithm == 'gp':
            return self.model.predict(features, eval_MSE=True)
        return self.model.predict(features) # value predicted in dt seconds. 


    def crossValidation(self, n): 
        kf = KFold(len(self.trainX), n_folds = n)
        total_error = 0
        predictions = {}
        if self.algorithm != 'gp':
            for train,test in kf: 
                this_x = []
                this_y = []
                for i in train: 
                    this_x.append(self.trainX[i])
                    this_y.append(self.trainY[i])
                reg = self.model_for_algorithm()
                reg.fit(this_x, this_y)
                for test_i in test: 
                    predicted = reg.predict(self.trainX[test_i])
                    predictions[test_i] = predicted
                    squared_error = (predicted - self.trainY[test_i])**2
                total_error += squared_error
            self.count_accuracy(predictions)
            return total_error / len(self.trainX), predictions
        else:
            for train_idx, test_idx in kf:
                X_train = self.trainX[train_idx]
                y_train = self.trainY[train_idx]
                gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
                gp.fit(X_train, y_train)
                for test_i in test_idx:
                    predicted, sigma2 = gp.predict(self.trainX[test_i], eval_MSE=True)
                    predictions[test_i] = (predicted, sigma2, self.trainY[test_i])
                    sigma = np.sqrt(sigma2)
                    if self.trainY[test_i] > predicted + 1.96 * sigma or self.trainY[test_i] < predicted - 1.96 * sigma:
                        total_error += 1
            return total_error / float(len(self.trainX)), predictions


    def count_accuracy(self, predictions):
        false_neg, true_neg, false_pos, true_pos = 0,0,0,0
        for i in range(0,len(self.trainY)): 
            if self.trainY[i] < 0 and predictions[i] < 0: 
                true_neg += 1
            elif self.trainY[i] > 0 and predictions[i] > 0: 
                true_pos += 1
            elif predictions[i] > 0: 
                false_pos += 1 
            elif predictions[i] < 0: 
                false_neg += 1 
        print 'True pos', true_pos, 'True neg', true_neg, 'False pos', false_pos, 'false_neg', false_neg


	def count_accuracy(self, predictions):
		false_neg, true_neg, false_pos, true_pos = 0,0,0,0
		for i in range(0,len(self.trainY)): 
			if self.trainY[i] < 0 and predictions[i] < 0: 
				true_neg += 1
			elif self.trainY[i] > 0 and predictions[i] > 0: 
				true_pos += 1
			elif predictions[i] > 0: 
				false_pos += 1 
			elif predictions[i] < 0: 
				false_neg += 1 
		print 'True pos', true_pos, 'True neg', true_neg, 'False pos', false_pos, 'false_neg', false_neg
