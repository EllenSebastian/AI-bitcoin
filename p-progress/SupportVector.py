import numpy as np
from sklearn.svm import SVR
import pylab as pl
import pickle, math, random, operator
from NeuralNetwork import Window
import pdb

# data from Oct 20, 2014, backwards, every hour
endTimeStamp= 1413763200

# get bitcoin price data	
execfile('dataFetcher.py')
priceData = pickle.load(open('../data/bitcoin_prices.pickle'))


class SupportVectorMachine:

	# nntype is ff, elman
	def __init__(self, windowSize = 10, numFeatures = 100, numDataPoints = 1000, frequency = 3600):
		self.windowSize = windowSize
		self.numFeatures = numFeatures
		self.endTimeStamp = endTimeStamp
		self.numDataPoints = numDataPoints
		self.frequency = frequency
		self.priceData = aggregated_prices(priceData, self.endTimeStamp, self.numDataPoints, self.frequency)

	def toPercentChange(self):
		""" takes in an list of price data and returns a list of percentage change price data """
		percentChange = list()
		for i in range(1, len(self.priceData)):
			percentChange.append((self.priceData[i] - self.priceData[i - 1])/self.priceData[i - 1])
		return percentChange

	def simulate(self, C=1.0, epsilon=0.1, kernel='rbf', degree=3.0):
		"""
			see documentation at http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
			to find out what each of the parameters means
		"""

		# list holding all our predictions
		predictedPercentChanges = list() 

		# list holding all the actuall percentage changes
		actualPercentChanges = list()

		# list holding the predicted prices
		predictedPrices = list()

		percentChangePriceData = self.toPercentChange()
		inputVector = Window(self.numFeatures)
		targetVector = Window(self.numFeatures)
		featureVector = Window(self.windowSize)

		# other parameter options available see documentation
		clf = SVR(C=C, epsilon=epsilon)


		# iterate over the price data to len(data) - 2 to avoid overflow because we predict step + 2 at each iteration
		for step in range(len(percentChangePriceData) - 2):

			featureVector.append(percentChangePriceData[step])
			if featureVector.isFull():
				inputVector.append(list(featureVector))
				targetVector.append(percentChangePriceData[step + 1])
				if inputVector.isFull() and targetVector.isFull():

					# we have enough input and target vectors to fit the SVR
					inputs = np.array(inputVector).reshape(self.numFeatures, self.windowSize)
					targets = np.array(targetVector)

					clf.fit(inputs, targets)

					# predict next time step
					testFeatureVector = featureVector[1:] + [percentChangePriceData[step + 1]]	
					out = clf.predict(testFeatureVector)
					predictedPercentChanges.append(out[0])
					predictedPrices.append((out[0] * self.priceData[step + 2]) + self.priceData[step + 2])
					actualPercentChanges.append(percentChangePriceData[step + 2])
					print "Done with %f of the process" % (float(step)/len(percentChangePriceData) * 100)

		pl.figure(1)
		pl.title("Price Data")
		pl.subplot(211)
		pl.plot(range(len(predictedPrices)), self.priceData[len(self.priceData) - len(predictedPrices) :], 'b--')
		pl.subplot(212)
		pl.plot(range(len(predictedPrices)), predictedPrices, 'r--')

		def graphData(predictedPercentChanges, actualPercentChanges):

			def graphError(predictedPercentChanges, actualPercentChanges):

				# considering error and only considering it as error when the signs are different
				def computeSignedError(pred, actual):
					if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
						return 0
					else :
						error =  abs(pred - actual)
						# print 'pred: {0}, actual: {1}, error: {2}'.format(pred, actual, error)
						return error

				signedError = map(lambda pred, actual: computeSignedError(pred, actual), predictedPercentChanges, actualPercentChanges)
				pl.figure(2)
				pl.title("Error")
				pl.subplot(211)
				pl.plot(signedError)
				pl.xlabel('Time step')
				pl.ylabel('Error (0 if signs are same and normal error if signs are different)')

				pl.figure(3)
				pl.title("Actual vs Predictions")
				pl.subplot(211)
				pl.plot(range(len(predictedPercentChanges)), predictedPercentChanges, 'ro', \
					range(len(actualPercentChanges)), actualPercentChanges, 'bs')

			def percentageCorrect(predictions, actuals):
				numCorrect = 0
				for i in range(len(predictions)):
					if (predictions[i] > 0 and actuals[i] > 0) or (predictions[i] < 0 and actuals[i] < 0):
						numCorrect = numCorrect + 1
				return numCorrect / float(len(predictions)) 
			print "The percentage correct is %f." % (percentageCorrect(predictedPercentChanges, actualPercentChanges))
			graphError(predictedPercentChanges, actualPercentChanges)		

		graphData(predictedPercentChanges, actualPercentChanges)
		pl.show()

def main():

	print "Starting Support Vector Machine Simulations"

	svr = SupportVectorMachine()
	svr.simulate()

	# # larger window
	# neuralNetwork2 = NeuralNetwork(48, 10, 200)
	# neuralNetwork2.simulate()

	# # large window
	# neuralNetwork3 = NeuralNetwork(32, 10, 200)
	# neuralNetwork3.simulate()

	# # day sized window
	# neuralNetwork4 = NeuralNetwork(24, 10, 200)
	# neuralNetwork4.simulate()

	# half a day sized window
	# neuralNetwork5 = NeuralNetwork(12, 10, 200)
	# neuralNetwork5.simulate()

	# quarter of a day sized window
	#neuralNetwork6 = NeuralNetwork(6, 10, 200)
	#neuralNetwork6.simulate()


if __name__ == "__main__": 
	main()

