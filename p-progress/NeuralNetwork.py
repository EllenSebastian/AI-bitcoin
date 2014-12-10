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


class NeuralNetwork:

	# nntype is ff, elman
	def __init__(self, endTimeStamp, windowSize = 10, numFeatures = 100, numDataPoints = 1000, frequency = 3600, nnType = 'ff', whichData=['price']):

		self.windowSize = windowSize
		self.numFeatures = numFeatures
		self.endTimeStamp = endTimeStamp
		self.numDataPoints = numDataPoints
		self.frequency = frequency

		if 'price' in whichData:		
			# get bitcoin price data	
			priceData = pickle.load(open('../data/bitcoin_prices.pickle'))
			self.listPriceData, self.mappedPriceData, mappedListData, self.timeRange = aggregated_data(priceData, self.endTimeStamp, self.numDataPoints, self.frequency)
			self.mappedListData = sorted(mappedListData, key=lambda elem: elem[0]) 
			
		self.type = nnType

		if self.type == 'elman':
			self.net = nl.net.newelm([[-1, 1] for i in range(self.windowSize)], [5, 1])
			pdb.set_trace()
			self.net.layers[0].initf = nl.init.InitRand([-10, 10], 'wb')
			self.net.layers[1].initf= nl.init.InitRand([-10, 10], 'wb')
			self.net.init()
		else:
			self.net = nl.net.newff([[-1, 1] for i in range(self.windowSize)], [20, 10, 5, 1])

		self.percentChangePriceData = self.toPercentChange(self.listPriceData)
		self.percentChangeOfPercentChangePriceData = self.toPercentChange(self.percentChangePriceData)

	def toPercentChange(self, data):
		""" takes in an list of price data and returns a list of percentage change price data """
		percentChange = list()
		for i in range(1, len(data)):
			percentChange.append((data[i] - data[i - 1])/data[i - 1])
		return percentChange

	def predictPrice(self, time_stamp, n):
		"""
			given a time_stamp and an n
			returns two lists of length containing the predicted prices and percentChanges
			return (percentChanges, predictedPrices) 
		"""
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
		percentChanges = {}
		inputVector = Window(self.numFeatures)
		targetVector = Window(self.numFeatures)
		featureVector = Window(self.windowSize)
		step = 0

		while (step < len(percentChangePriceData) and len(percentChanges.keys()) < n):
			cur_ts = time_stamp + step * self.frequency
			featureVector.append(percentChangePriceData[step])
			if featureVector.isFull():
				inputVector.append(list(featureVector))
				targetVector.append(percentChangePriceData[step + 1])
				if inputVector.isFull() and targetVector.isFull():

					inputs = np.array(inputVector).reshape(self.numFeatures, self.windowSize)
					targets = np.array(targetVector).reshape(self.numFeatures, 1)
					err = self.net.train(inputs, targets, goal = 0.01)

					# predict next step
					testFeatureVector = featureVector[1:] + [percentChangePriceData[step + 1]]
					out = self.net.sim([np.array(testFeatureVector)])
					percentChanges[cur_ts] = out[0][0]
					percentChangePriceData.append(out[0][0])
			step += 1
		predictedPrices = {}
		lastPrice = newPriceData[len(priceData) - 1]
		for p in percentChanges.keys():
			newPrice = (percentChanges[p] * lastPrice) + lastPrice
			predictedPrices[p] = newPrice
			lastPrice = newPrice  
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
		for step in range(len(self.percentChangeOfPercentChangePriceData) - 2):
			featureVector.append(self.percentChangeOfPercentChangePriceData[step])
			if featureVector.isFull():
				inputVector.append(list(featureVector))
				targetVector.append(self.percentChangeOfPercentChangePriceData[step + 1])
				if inputVector.isFull() and targetVector.isFull():

					inputs = np.array(inputVector).reshape(self.numFeatures, self.windowSize)
					targets = np.array(targetVector).reshape(self.numFeatures, 1)
					err = self.net.train(inputs, targets, goal = 0.01)

					testFeatureVector = featureVector[1:] + [self.percentChangePriceData[step + 1]]	
					out = self.net.sim([np.array(testFeatureVector)])

					predictedPPChanges.append(out[0][0])
					actualPPChanges.append(self.percentChangeOfPercentChangePriceData[step + 2])
					
					percentChangePrediction = (out[0][0] * self.percentChangePriceData[step + 2]) + self.percentChangePriceData[step + 2] 
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
		# list holding all our predictions
		predictedPercentChanges = list() 

		# list holding all the actuall percentage changes
		actualPercentChanges = list()

		# list holding the predicted prices
		predictedPrices = list()

		inputVector = Window(self.numFeatures)
		targetVector = Window(self.numFeatures)
		featureVector = Window(self.windowSize)

		# [5,3,1]: 0.520810
		# [10,5,1]: 0.546682
		# [20,5,1]: 0.534308
		# [20,10,5,1]: 0.509561

		# iterate over the price data to len(data) - 2 to avoid overflow because we predict step + 2 at each iteration
		for step in range(len(self.percentChangePriceData) - 2):

			featureVector.append(self.percentChangePriceData[step])
			if featureVector.isFull():
				inputVector.append(list(featureVector))
				targetVector.append(self.percentChangePriceData[step + 1])
				if inputVector.isFull() and targetVector.isFull():

					# we have enough input and target vectors to train the neural network 
					# create a 2 layer forward feed neural network
					
					inputs = np.array(inputVector).reshape(self.numFeatures, self.windowSize)
					targets = np.array(targetVector).reshape(self.numFeatures, 1)
					err = self.net.train(inputs, targets, goal = 0.01)
					
					# predict next time step
					testFeatureVector = featureVector[1:] + [self.percentChangePriceData[step + 1]]	
					out = self.net.sim([np.array(testFeatureVector)])

					predictedPercentChanges.append(out[0][0])
					predictedPrices.append((out[0][0] * self.listPriceData[step + 2]) + self.listPriceData[step + 2])
					actualPercentChanges.append(self.percentChangePriceData[step + 2])
					print "Done with %f of the process" % (float(step)/len(self.percentChangePriceData) * 100)

		pl.figure(1)
		pl.title("Price Data")
		pl.subplot(211)
		pl.plot(range(len(predictedPrices)), self.listPriceData[len(self.listPriceData) - len(predictedPrices) :], 'b--')
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

	print "Starting Neural Network Simulations"

	basicNeuralNetwork = NeuralNetwork(1413230400, 6, 10, 200)
	basicNeuralNetwork.simulateWithSecondDerivative()

	# neuralNetwork3 = NeuralNetwork(32)
	# neuralNetwork3.simulate()

	# # vary window size
	# neuralNetwork1 = NeuralNetwork(20, 10, 500, 60)
	# neuralNetwork1.simulate()

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


	# try predicting with this time_stamp 1413230400
	# neuralNetwork6 = NeuralNetwork(6, 10, 200)
	# neuralNetwork6.predictPrice(1413230400, 10)


if __name__ == "__main__": 
	main()
