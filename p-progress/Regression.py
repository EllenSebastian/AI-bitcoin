# extract features for regression
import numpy, pickle, random, pdb
# features for this test : 
N_EXAMPLES = 1000
DAYS_FOR_FEATURES = 30 # number of days to look in the past for non-price features. 
dt = 60 
import pickle, math, random

execfile('dataFetcher.py')
prices = pickle.load(open('../data/bitcoin_prices.pickle') )
transactions_per_minute = pickle.load(open('../data/transactions_per_minute.pickle') )
sorted_timestamps = sorted(prices.keys(), reverse=True)

non_price_inputs =  ['avg-confirmation-time.txt', 'estimated-transaction-volume.txt', 'my-wallet-transaction-volume.txt', 'total-bitcoins.txt', 
             'bitcoin-days-destroyed-cumulative.txt','hash-rate.txt', 'n-orphaned-blocks.txt','trade-volume.txt', 'bitcoin-days-destroyed.txt','market-cap.txt', 
             'n-transactions-excluding-popular.txt','transaction-fees.txt', 'blocks-size.txt','n-transactions-per-block.txt', 'tx-trade-ratio.txt', 
             'cost-per-transaction.txt','miners-revenue.txt', 'n-transactions.txt', 'difficulty.txt','my-wallet-n-tx.txt', 'n-unique-addresses.txt', 
             'estimated-transaction-volume-usd.txt', 'my-wallet-n-users.txt', 'output-volume.txt']

data = {}
for f in non_price_inputs: 
    data[f] = read_data('../data/' + f)

possible_train = []

train_examples = transactions_per_minute.keys()
for i in xrange(len(sorted_timestamps)-1):
	if sorted_timestamps[i] - sorted_timestamps[i+1] == 60 and sorted_timestamps[i] in transactions_per_minute: 
		possible_train.append(sorted_timestamps[i])
# end up with 98707 places to choose

train_examples = random.sample(possible_train, N_EXAMPLES)

all_features = []
all_Y = []

def features_for_ts(train_ts): 
	last_60 = transactions_per_minute[train_ts]
	n_sell = sum([int(x[1] == 'buy') for x in last_60])
	n_buy = float(sum([int(x[1] == 'sell') for x in last_60]))
	amt_sell = numpy.mean([float(x[0]) for x in last_60 if x[1] == 'sell'])
	if n_sell > 0 and n_buy > 0: 
		log_buy_sell_ratio = math.log(float(n_buy) / n_sell) 
	elif n_sell is 0:
		log_buy_sell_ratio = math.log(60)
	else:
		log_buy_sell_ratio = math.log(1.0/60)
	#ticker = get_ticker()['ticker'] # HISTORICAL?
	features = [log_buy_sell_ratio]# TODO, float(ticker['sell']), float(ticker['buy']), float(ticker['last']), float(ticker['vol']), float(ticker['high']), float(ticker['low'])]
	assert len(features) == 1
	# 1 hour of minutes
	for i in range(0, 60): 
		features += [prices[train_ts - i*60]]
	assert len(features) == 61
	# 1 day of hours
	features += aggregated_prices(prices, train_ts - (60 * 60), 24, 60 * 60)
	assert len(features) == 61 + 24 
	# 60 days of days
	#pdb.set_trace()
	features += aggregated_prices(prices, train_ts - (60 * 60 * 25), 60, 60 * 60 * 24)
	assert len(features) == 61 + 24 + 60  
	start_datetime = datetime.datetime.fromtimestamp(train_ts)
	end_day = datetime.date(start_datetime.year, start_datetime.month, start_datetime.day)
	start_day = end_day - datetime.timedelta(days=DAYS_FOR_FEATURES)
	for file in non_price_inputs:
		cur_features = make_feature_vector_from_file(data[file], start_day, end_day) 
		features += cur_features 
	return features	

all_Y = []
all_features = []
for train_ts in train_examples:
	print train_ts
	features = features_for_ts(train_ts)
	while None in features: 
		pdb.set_trace()
		while train_ts in train_examples: 
			train_ts = random.choice(possible_train)
		features = features_for_ts(train_ts)
	all_features.append(features)
	all_Y.append((prices[train_ts + dt] - prices[train_ts]) / float(prices[train_ts]))
	# average price over the last 60 minutes, last 24 hours, last 60 days
	if train_ts == train_examples[0]: 
		pdb.set_trace()
	if len(all_Y) >= 1000: 
		break 

linear_model_ = PricePredictor(all_features, all_Y, 'linear')
linear_err, linear_predictions= linear_model_.crossValidation(10)
# True pos 247 True neg 234 False pos 275 false_neg 244


bayesian_model = PricePredictor(all_features, all_Y, 'BayesianRidge')
bayes_err, bayes_predictions= bayesian_model.crossValidation(10)
# True pos 241 True neg 261 False pos 244 false_neg 254


ridge_model = PricePredictor(all_features, all_Y, 'ridge')
ridge_err, ridge_predictions= ridge_model.crossValidation(10)
# True pos 247 True neg 234 False pos 275 false_neg 244

logistic_model = PricePredictor(all_features, all_Y, 'logistic') # does not finish
err, predictions= pp_linear.crossValidation(10)

perceptron_model = PricePredictor(all_features, all_Y, 'perceptron')
perceprton_err, perceptron_predictions= pp_linear.crossValidation(10)


pred = []
for i in range(0,1000):
	pred.append(linear_predictions[i])

import matplotlib.pyplot as plt
plt.scatter(all_Y, pred)
plt.xlabel('Actual Delta P values')
plt.ylabel('Predicted Delta P values')
plt.title('Predicted vs actual Delta P values for Linear Regression')
plt.show()
