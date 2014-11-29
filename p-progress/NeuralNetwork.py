import numpy as np
import neurolab as nl
import pylab as pl
import pickle, math, random, operator
import pdb

# data from Oct 20, 2014, backwards, every hour
endTimeStamp= 1413763200

# get bitcoin price data	
execfile('dataFetcher.py')
priceData = pickle.load(open('../data/bitcoin_prices.pickle'))

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

	def __init__(self, windowSize = 10, numFeatures = 100, numDataPoints = 1000, frequency = 3600):

		self.windowSize = windowSize
		self.numFeatures = numFeatures
		self.endTimeStamp = endTimeStamp
		self.numDataPoints = numDataPoints
		self.frequency = frequency
		self.priceData = prices = aggregated_prices(priceData, self.endTimeStamp, self.numDataPoints, self.frequency)


	def toPercentChange(self):
		""" takes in an list of price data and returns a list of percentage change price data """
		percentChange = list()
		for i in range(1, len(self.priceData)):
			percentChange.append((self.priceData[i] - self.priceData[i - 1])/self.priceData[i - 1])
		return percentChange

	def simulate(self):
		# list holding all our predictions
		predictions = list() 

		# list holding all the actuall percentage changes
		actuals = list()

		percentChangePriceData = self.toPercentChange()
		inputVector = Window(self.numFeatures)
		targetVector = Window(self.numFeatures)
		featureVector = Window(self.windowSize)

		net = nl.net.newff([[-1, 1] for i in range(self.windowSize)], [self.numFeatures, 1])

		# iterate over the price data to len(data) - 2 to avoid overflow because we predict step + 2 at each iteration
		for step in range(len(percentChangePriceData) - 2):

			featureVector.append(percentChangePriceData[step])
			if featureVector.isFull():
				inputVector.append(list(featureVector))
				targetVector.append(percentChangePriceData[step + 1])
				if inputVector.isFull() and targetVector.isFull():
					

					# we have enough input and target vectors to train the neural network 
					# create a 2 layer forward feed neural network
					
					inputs = np.array(inputVector).reshape(self.numFeatures, self.windowSize)
					targets = np.array(targetVector).reshape(self.numFeatures, 1)
					err = net.train(inputs, targets, epochs = 500, goal = 0.005)
					
					# predict next time step
					testFeatureVector = featureVector[1:] + [percentChangePriceData[step + 1]]	
					out = net.sim([np.array(testFeatureVector)])
					predictions.append(out[0][0])
					actuals.append(percentChangePriceData[step + 2])
					print "Done with %f of the process" % (float(step)/len(percentChangePriceData) * 100) 

		def graphData(predictions, actuals):

			def graphError(predictions, actuals):
				# considering error and only considering it as error when the signs are different
				def computeSignedError(pred, actual):
					if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
						return 0
					else :
						return abs(pred - actual / actual) * 100

				signedError = map(lambda pred, actual: computeSignedError(pred, actual), predictions, actuals)
				pl.figure(1)
				pl.title("Error")
				pl.subplot(211)
				pl.plot(signedError)
				pl.xlabel('Time step')
				pl.ylabel('Error (0 if signs are same and normal error if signs are different)')

				pl.figure(2)
				pl.title("Actual vs Predictions")
				pl.subplot(211)
				pl.plot(range(len(predictions)), predictions, 'r--', range(len(actuals)), actuals, 'bs')
				pl.show()

			def percentageCorrect(predictions, actuals):
				numCorrect = 0
				for i in range(len(predictions)):
					if (predictions[i] > 0 and actuals[i] > 0) or (predictions[i] < 0 and actuals[i] < 0):
						numCorrect = numCorrect + 1
				return numCorrect / float(len(predictions)) 

			print "The percentage correct is %f." % (percentageCorrect(predictions, actuals))
			graphError(predictions, actuals)		

		graphData(predictions, actuals)

		
def main():

	print "Starting Neural Network Simulations"
	# basicNeuralNetwork = NeuralNetwork()
	# basicNeuralNetwork.simulate()

	# vary window size
	neuralNetwork1 = NeuralNetwork(60, 10, 200)
	neuralNetwork1.simulate()

	# larger window
	neuralNetwork2 = NeuralNetwork(48, 10, 200)
	neuralNetwork2.simulate()

	# large window
	neuralNetwork3 = NeuralNetwork(32, 10, 200)
	neuralNetwork3.simulate()

	# day sized window
	neuralNetwork4 = NeuralNetwork(24, 10, 200)
	neuralNetwork4.simulate()

	# half a day sized window
	neuralNetwork5 = NeuralNetwork(12, 10, 200)
	neuralNetwork5.simulate()

	# quarter of a day sized window
	neuralNetwork6 = NeuralNetwork(6, 10, 200)
	neuralNetwork6.simulate()

	# #simulating the prediction algorithm starting from Oct 20, 2014, every hour
	# hour_10f_10w = simulate(prices, 10, 10, 1413763200, 10000, 3600)
	# # tp 2681 tn 2359 fp 2281 fn 2499

	# hour_100f_10w = simulate(prices, 100, 10, 1413763200, 10000, 3600)
	# # tp 25391 tn 21679 fp 20906 fn 21224

	# # ASSERTION ERROR input.shape[1] == net.ci
	# min_10f_100w = simulate(prices, 10, 100, 1413763200, 10000, 60)

	# # ASSERTION ERROR input.shape[1] == net.ci
	# hour_100f_100w = simulate(prices, 100, 100, 1413763200, 10000, 3600)
	# # tp 25804 tn 21390 fp 21195 fn 20811

	# hour_200f_200w = simulate(prices, 200, 200, 1413763200, 10000, 3600)
	# # tp 25034 tn 21707 fp 20878 fn 21581

	# hour_300f_100w = simulate(prices, 300, 100, 1413763200, 10000, 3600)
	# # tp 24725 tn 22141 fp 20444 fn 21890

if __name__ == "__main__": main()

