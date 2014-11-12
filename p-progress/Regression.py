# extract features for regression
import numpy, pickle, random, pdb
# features for this test : 
N_EXAMPLES = 1000

import pickle, math, random

execfile('dataFetcher.py')
prices = pickle.load(open('../data/bitcoin_prices.pickle') )
transactions_per_minute = pickle.load(open('../data/transactions_per_minute.pickle') )
sorted_timestamps = sorted(prices.keys(), reverse=True)

possible_train = []

train_examples = transactions_per_minute.keys()
for i in xrange(len(sorted_timestamps)-1):
	if sorted_timestamps[i] - sorted_timestamps[i+1] == 60 and sorted_timestamps[i] in transactions_per_minute: 
		possible_train.append(sorted_timestamps[i])
# end up with 98707 places to choose

train_examples = random.sample(possible_train, N_EXAMPLES)

all_features = []
all_Y = []
for train_ts in train_examples:
	print train_ts
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
	all_features.append(features)
	all_Y.append(prices[train_ts + 60])
	# average price over the last 60 minutes, last 24 hours, last 60 days


pp_linear = PricePredictor.PricePredictor(all_features, all_Y, 'linear')
