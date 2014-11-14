import numpy as np
import neurolab as nl
import pickle, math, random, operator

# has to be greater than 3
window_size = 3
num_features = 10

execfile('dataFetcher.py')
prices = pickle.load(open('../data/bitcoin_prices.pickle') )

#temporally_sorted_prices = sorted(prices.items(), key=operator.itemgetter(0))

class Window(list):
	def __init__(self, size):
		self.size = size

	def append(self, item):
		super(Window, self).append(item) 
		if len(self) > self.size:
			super(Window, self).pop(0)

# def createTrainExamples(index, temporally_sorted_prices, window_size, num_features):
# 	for i in range(num_features):
# 		print temporally_sorted_prices[index - window_size - i : index - i]
# 		print i 
def normalize(inputs):
	normalized_inputs = list()
	for i in range(len(inputs)):
		normed = list()
		for j in range(1, len(inputs[i])):
			normed.append((inputs[i][j] - inputs[i][j - 1]) / inputs[i][j])
		normalized_inputs.append(normed)
	return normalized_inputs

def createTrainExamples(normalized_inputs):
	normalized_inputs = np.array(normalized_inputs)
	feature_length = window_size - 1
	targets = normalized_inputs[:, feature_length - 1]
	inputs = normalized_inputs[:, 0:feature_length]
	return inputs.reshape(len(normalized_inputs), feature_length), targets.reshape(len(normalized_inputs), 1)

def simulate(priceData, num_features, window_size, end_timestamp, num_aggregates, aggregation):
	prices = aggregated_prices(priceData, end_timestamp, num_aggregates, aggregation)
	window = Window(window_size)
	inputs_and_targets = Window(num_features)
	for index in range(100):
		window.append(prices[index])
		if len(window) == window_size:
			inputs_and_targets.append(window[0:len(window)])
			if len(inputs_and_targets) == num_features:
				normalized_inputs_and_targets = normalize(inputs_and_targets)
				
				# train artifical neural net
				pdb.set_trace()
				net = nl.net.newff([[-1, 1] for i in range(num_features)], [num_features, 1])
				inputs, targets = createTrainExamples(normalized_inputs_and_targets)
				err = net.train(inputs, targets, show=15)
				
				# predict the next value
				# update error

#simulating the prediction algorithm starting from Oct 20, 2014, every hour
simulate(prices, num_features, window_size, 1413763200, 10000, 3600)