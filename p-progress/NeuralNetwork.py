import numpy as np
import neurolab as nl
import pickle, math, random, operator

# has to be greater than 3
window_size = 10
num_features = 100

execfile('dataFetcher.py')
prices = pickle.load(open('../data/bitcoin_prices.pickle') )

#temporally_sorted_prices = sorted(prices.items(), key=operator.itemgetter(0))

class Window(list):
	def __init__(self, size=0, arr=[]):
		if size == 0: 	
			self.size = len(arr)
		else: 
			self.size = size
		for i in arr: 
			super(Window, self).append(i)
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


def createTrainExamples(normalized_inputs, window_size ):
	normalized_inputs = np.array(normalized_inputs)
	feature_length = window_size - 1
	targets = normalized_inputs[:, feature_length - 2]
	inputs = normalized_inputs[:, 0:feature_length]
	return inputs.reshape(len(normalized_inputs), feature_length), targets.reshape(len(normalized_inputs), 1)


def simulate(priceData, num_features, window_size, end_timestamp, num_aggregates, aggregation):
	fn, fp, tn, tp = 0,0,0,0
	prices = aggregated_prices(priceData, end_timestamp, num_aggregates, aggregation)
	window = Window(window_size)
	inputs_and_targets = Window(num_features)
	errs = []
	for index in range(200):
		#pdb.set_trace()
		window.append(prices[index])
		if len(window) == window_size:
			inputs_and_targets.append(window[0:len(window)])
			if len(inputs_and_targets) == num_features:
				normalized_inputs_and_targets = normalize(inputs_and_targets)			
				# train artifical neural net
				# should we reinitialize the network every time? or just add more trainig to it?
				net = nl.net.newff([[-1, 1] for i in range(window_size - 1)], [num_features, 1])
				inputs, targets = createTrainExamples(normalized_inputs_and_targets, window_size)
				err = net.train(inputs, targets, show=15, epochs = 20)
				next_window = Window(len(window), window)
				next_window.append(prices[index + 1])
				test_input = normalize([next_window])
				out = net.sim(test_input)
				actualDP = (prices[index + 2] - prices[index + 1]) / prices[index + 1]
				errs.append((out[0][0],actualDP))
				print errs[len(errs) - 1]
				if out[0][0] < 0 and actualDP < 0: 
					tn += 1
				elif out[0][0] < 0 and actualDP >= 0:
					fn += 1  
				elif out[0][0] > 0 and actualDP < 0: 
					fp += 1
				elif out[0][0] >= 0 and actualDP >= 0:
					tp += 1
				else: 
					print out[0][0], actualDP
	print 'tp',tp ,'tn', tn, 'fp',fp, 'fn', fn
	return errs, tp, tn, tp, fn
				# predict the next value
				# update error


#simulating the prediction algorithm starting from Oct 20, 2014, every hour
hour_10f_10w = simulate(prices, 10, 10, 1413763200, 10000, 3600)
# tp 279 tn 259 fp 206 fn 238
# 0.5478615071283096
window_size = 10
num_features = 100
hour_100f_10w = simulate(prices, 100, 10, 1413763200, 10000, 3600)
# tp 243 tn 226 fp 201 fn 222
# 0.6777456647398844


window_size = 100
num_features = 10
min_10f_100w = simulate(prices, 10, 100, 1413763200, 10000, 60)
# 0.5230596175478065

hour_100f_25w = simulate(prices, 100, 24, 1413763200, 10000, 3600)
#tp 250 tn 232 fp 187 fn 209
#0.5489749430523918

# ASSERTION ERROR input.shape[1] == net.ci
hour_100f_25w = simulate(prices, 100, 11, 1413763200, 10000, 3600)
# tp 25804 tn 21390 fp 21195 fn 20811

hour_200f_200w = simulate(prices, 200, 200, 1413763200, 10000, 3600)
# tp 25034 tn 21707 fp 20878 fn 21581

hour_300f_100w = simulate(prices, 300, 100, 1413763200, 10000, 3600)
# tp 24725 tn 22141 fp 20444 fn 21890


def plot_predictions(simulate_out):
	pred = [res[0] for res in simulate_out[0]]
	actual = [res[0] for res in simulate_out[0]]
	import matplotlib.pyplot as plt
	plt.scatter(pred, actual)
	plt.xlabel('Actual Delta P values')
	plt.ylabel('Predicted Delta P values')
	plt.title('Predicted vs actual Delta P values for Bayesian Ridge Regression')
	plt.show()
