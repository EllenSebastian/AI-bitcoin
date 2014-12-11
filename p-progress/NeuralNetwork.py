import numpy as np
import neurolab as nl
import pylab as pl
import pickle, math, random, operator
import pdb

execfile('dataFetcher.py')
#priceData = pickle.load(open('../data/bitcoin_prices.pickle'))
# data from Oct 20, 2014, backwards, every hour
#endTimeStamp= 1413763200
class Window(list):

	def __init__(self, size):
		self.size = size

	def append(self, item):
		super(Window, self).append(item) 
		if len(self) > self.size:
			super(Window, self).pop(0)

	def isFull(self):
		return len(self) == self.size

priceData = pickle.load(open('../data/bitcoin_prices.pickle'))


class NeuralNetwork:

	def __init__(self, endTimeStamp, windowSize = 10, numFeatures = 100, numDataPoints = 1000, frequency = 3600, nnType = 'ff', whichData=['price'], normalize=True):
	# nntype is ff, elman

		self.windowSize = windowSize
		self.numFeatures = numFeatures
		self.endTimeStamp = endTimeStamp
		self.numDataPoints = numDataPoints
		self.frequency = frequency
		self.normalize = normalize

		if 'price' in whichData:		
			# get bitcoin price data	
			self.listPriceData, self.mappedPriceData, mappedListData, self.timeRange = aggregated_data(priceData, self.endTimeStamp, self.numDataPoints, self.frequency)
			self.mappedListData = sorted(mappedListData, key=lambda elem: elem[0]) 
			
		self.type = nnType
		
		def normalizeData(X, a, b):
			"""
				pass in list X and return a list normalized between a and b
			"""
			maximum = max(X)
			minimum = min(X)
			r = maximum - minimum
			n = b - a
			return map(lambda x: a + (((x - minimum) / r) * n), X)

		def refactor(inp, minimum, maximum, a, b, normalize):
						if normalize:
							return (maximum - minimum)*((inp - a)/(b - a)) + minimum
						return inp
		self.normalizeData = normalizeData
		self.refactor = refactor

		self.a = -10000
		self.b = 10000

		self.percentChangePriceData = self.toPercentChange(self.listPriceData)
		self.percentChangePriceDataMax = max(self.percentChangePriceData)
		self.percentChangePriceDataMin = min(self.percentChangePriceData)
		self.percentChangeOfPercentChangePriceData = self.toPercentChange(self.percentChangePriceData)
		self.percentChangeOfPercentChangePriceDataMax = max(self.percentChangeOfPercentChangePriceData)
		self.percentChangeOfPercentChangePriceDataMin = min(self.percentChangeOfPercentChangePriceData)

		if normalize:
			self.normalizedPercentChangePriceData = self.normalizeData(self.percentChangePriceData, self.a, self.b)
			self.normalizedPercentChangeOfPercentChangePriceData = self.normalizeData(self.percentChangeOfPercentChangePriceData, self.a, self.b)


		if self.type == 'elman':
			self.net = nl.net.newelm([[-1, 1] for i in range(self.windowSize)], [5, 1])
			pdb.set_trace()
			self.net.layers[0].initf = nl.init.InitRand([-10, 10], 'wb')
			self.net.layers[1].initf= nl.init.InitRand([-10, 10], 'wb')
			self.net.init()
		else:
			self.net = nl.net.newff([[self.a, self.b] for i in range(self.windowSize)], [self.numFeatures, 1])


	def toPercentChange(self, data):
		""" takes in an list of price data and returns a list of percentage change price data """
		percentChange = list()
		for i in range(1, len(data)):
			percentChange.append((data[i] - data[i - 1])/data[i - 1])
		return percentChange

	def predictPrice(self, time_stamp, n = 1):
		"""
			given a time_stamp and an n
			returns two lists of length containing the predicted prices and percentChanges
			return (percentChanges, predictedPrices) 
			USE Second Derivative Method
		"""
		pdb.set_trace()

		price = self.mappedPriceData[time_stamp]
		index = self.mappedListData.index([time_stamp, price])
		numDataPoints = self.numFeatures + self.windowSize + 1
		startIndex = index - numDataPoints
		if (startIndex < 0):
			print "please use a later time stamp. there are not enough data points in the past to train on."
			return 
		priceData = self.mappedListData[startIndex : index]
		newPriceData = map(lambda elem: elem[1], priceData)
		percentChangePriceData = self.toPercentChange(newPriceData)
		percentChangePriceDataMax = max(percentChangePriceData)
		percentChangePriceDataMin = min(percentChangePriceData)
		normalizedPercentChangePriceData = self.normalizeData(percentChangePriceData, self.a, self.b)
		percentChanges = {}
		inputVector = Window(self.numFeatures)
		targetVector = Window(self.numFeatures)
		featureVector = Window(self.windowSize)
		step = 0

		while (step < len(normalizedPercentChangePriceData) and len(percentChanges.keys()) < n):
			cur_ts = time_stamp + step * self.frequency
			featureVector.append(normalizedPercentChangePriceData[step])
			if featureVector.isFull():
				inputVector.append(list(featureVector))
				targetVector.append(normalizedPercentChangePriceData[step + 1])
				if inputVector.isFull() and targetVector.isFull():

					inputs = np.array(inputVector).reshape(self.numFeatures, self.windowSize)
					targets = np.array(targetVector).reshape(self.numFeatures, 1)
					err = self.net.train(inputs, targets, goal = 0.01)

					# predict next step
					testFeatureVector = featureVector[1:] + [normalizedPercentChangePriceData[step + 1]]
					out = self.net.sim([np.array(testFeatureVector)])

					output = self.refactor(out[0][0], percentChangePriceDataMin, \
						percentChangePriceDataMax, self.a, self.b, self.normalize)
					percentChanges[cur_ts] = output
					normalizedPercentChangePriceData.append(out[0][0])
			step += 1

		predictedPrices = {}
		lastPrice = newPriceData[len(priceData) - 1]
		for p in percentChanges.keys():
			newPrice = (percentChanges[p] * lastPrice) + lastPrice
			predictedPrices[p] = newPrice
			lastPrice = newPrice  
		print percentChanges, predictedPrices
		return percentChanges, predictedPrices

	def simulateWithSecondDerivative(self):
		"""
		identical to simulateWithFirstDerivative
		"""

		# second derivate lists
		predictedPPChanges = list()
		actualPPChanges = list()

		# first derivative lists
		predictedPChanges = list()
		actualPChanges = list()

		predictedPrices = list()
		actualPrices = list()

		inputVector = Window(self.numFeatures)
		targetVector = Window(self.numFeatures)
		featureVector = Window(self.windowSize)


		# iterate over the price data to len(data) - 2 to avoid overflow because we predict step + 2 at each iteration
		for step in range(len(self.normalizedPercentChangeOfPercentChangePriceData) - 2):
			featureVector.append(self.normalizedPercentChangeOfPercentChangePriceData[step])
			if featureVector.isFull():
				inputVector.append(list(featureVector))
				targetVector.append(self.normalizedPercentChangeOfPercentChangePriceData[step + 1])
				if inputVector.isFull() and targetVector.isFull():

					inputs = np.array(inputVector).reshape(self.numFeatures, self.windowSize)
					targets = np.array(targetVector).reshape(self.numFeatures, 1)
					err = self.net.train(inputs, targets, goal = 0.01)

					testFeatureVector = featureVector[1:] + [self.normalizedPercentChangeOfPercentChangePriceData[step + 1]]	
					out = self.net.sim([np.array(testFeatureVector)])

					def toDecimal(inp):
						return inp / 100

					#pdb.set_trace()
					output = self.refactor(out[0][0], self.percentChangeOfPercentChangePriceDataMin, \
						self.percentChangeOfPercentChangePriceDataMax, self.a, self.b, self.normalize)

					predictedPPChanges.append(output)
					actualPPChanges.append(self.percentChangeOfPercentChangePriceData[step + 2])

	
					percentChangePrediction = (output * self.percentChangePriceData[step + 2]) + self.percentChangePriceData[step + 2] 
					predictedPChanges.append(percentChangePrediction)
					actualPChanges.append(self.percentChangePriceData[step + 3])
					
					pricePrediction = (percentChangePrediction * self.listPriceData[step + 3]) + self.listPriceData[step + 3]
					predictedPrices.append(pricePrediction)
					actualPrices.append(self.listPriceData[step + 4])

					print "Done with %f of the process" % (float(step)/len(self.percentChangeOfPercentChangePriceData) * 100)		

		pl.figure(1)
		pl.title("Price Data")
		pl.subplot(211)
		pl.plot(range(len(actualPrices)), actualPrices, 'b--')
		pl.subplot(212)
		pl.plot(range(len(predictedPrices)), predictedPrices, 'r--')

		def graphData(predictedPercentChanges, actualPercentChanges, title):

			def graphError(predictedPercentChanges, actualPercentChanges, title):
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
				pl.title( title + " Error")
				pl.subplot(211)
				pl.plot(signedError)
				pl.xlabel('Time step')
				pl.ylabel('Error (0 if signs are same and normal error if signs are different)')

				pl.figure(3)
				pl.title(title + " Actual vs Predictions")
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
			graphError(predictedPercentChanges, actualPercentChanges, title)	

		graphData(predictedPPChanges, actualPPChanges, "Second Derivative")
		#graphData(predictedPChanges, actualPChanges, "First Derivative")
		pl.show()



	def simulateWithFirstDerivative(self):

		def toDecimal(inp):
			return inp / 100

		# list holding all our predictions
		predictedPercentChanges = list() 

		# list holding all the actuall percentage changes
		actualPercentChanges = list()

		# list holding the predicted prices
		predictedPrices = list()
		actualPrices = list()

		inputVector = Window(self.numFeatures)
		targetVector = Window(self.numFeatures)
		featureVector = Window(self.windowSize)

		# [5,3,1]: 0.520810
		# [10,5,1]: 0.546682
		# [20,5,1]: 0.534308
		# [20,10,5,1]: 0.509561

		# iterate over the price data to len(data) - 2 to avoid overflow because we predict step + 2 at each iteration
		for step in range(len(self.normalizedPercentChangePriceData) - 2):

			featureVector.append(self.normalizedPercentChangePriceData[step])
			if featureVector.isFull():
				inputVector.append(list(featureVector))
				targetVector.append(self.normalizedPercentChangePriceData[step + 1])
				if inputVector.isFull() and targetVector.isFull():

					# we have enough input and target vectors to train the neural network 
					# create a 2 layer forward feed neural network
					inputs = np.array(inputVector).reshape(self.numFeatures, self.windowSize)
					targets = np.array(targetVector).reshape(self.numFeatures, 1)
					err = self.net.train(inputs, targets, goal = 0.01)

					# predict next time step
					testFeatureVector = featureVector[1:] + [self.normalizedPercentChangePriceData[step + 1]]	
					out = self.net.sim([np.array(testFeatureVector)])
					output = self.refactor(out[0][0], self.percentChangePriceDataMin, \
					 self.percentChangePriceDataMax, self.a, self.b, self.normalize)

					predictedPercentChanges.append(output)
					predictedPrices.append((output * self.listPriceData[step + 2]) + self.listPriceData[step + 2])
					actualPercentChanges.append(self.percentChangePriceData[step + 2])
					actualPrices.append(self.listPriceData[step + 3])
					print "Done with %f of the process" % (float(step)/len(self.percentChangePriceData) * 100)

		pl.figure(1)
		pl.title("Price Data")
		pl.subplot(211)
		pl.plot(range(len(actualPrices)), actualPrices, 'b--')
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
				pl.ylabel('Error (0 if signs are same and error if signs are different)')

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

	print "Starting Neural Network Simulations"

	basicNeuralNetwork1 = NeuralNetwork(1413230400, 6, 10, 500)
	#basicNeuralNetwork1.simulateWithSecondDerivative()
	#basicNeuralNetwork1.simulateWithFirstDerivative()

	# predict
	basicNeuralNetwork1.predictPrice(1411988400, 3)

	# basicNeuralNetwork1 = NeuralNetwork(1413230400, 12, 10, 500)
	# basicNeuralNetwork1.simulateWithSecondDerivative()
	# basicNeuralNetwork1.simulateWithFirstDerivative()

	# basicNeuralNetwork1 = NeuralNetwork(1413230400, 24, 10, 500)
	# basicNeuralNetwork1.simulateWithSecondDerivative()
	# basicNeuralNetwork1.simulateWithFirstDerivative()

	# basicNeuralNetwork1 = NeuralNetwork(1413230400, 32, 10, 500)
	# basicNeuralNetwork1.simulateWithSecondDerivative()
	# basicNeuralNetwork1.simulateWithFirstDerivative()

	# basicNeuralNetwork1 = NeuralNetwork(1413230400, 48, 10, 500)
	# basicNeuralNetwork1.simulateWithSecondDerivative()
	# basicNeuralNetwork1.simulateWithFirstDerivative()


if __name__ == "__main__": 
	main()
