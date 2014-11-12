# features for this test : 
import pickle, math, random

execfile('dataFetcher.py')
prices = pickle.load(open('bitcoin_prices.pickle') )
transactions_per_minute = pickle.load(open('transactions_per_minute.pickle') )


train_examples = transactions_per_minute.keys()
for i in xrange(len(sorted_timestamps)-1):
	if sorted_timestamps[i] - sorted_timestamps[i+1] == 60000: 
		possible_train.append(i)

		
sorted_timestamps = sorted(prices.keys(), reverse=True)
possible_train = []

train_examples = transactions_per_minute.keys()


for train_ts in train_examples:tmp = record_transactions_per_minute()

	last_60 = train_examples[train_ts]
	n_sell = sum([int(x[1] == 'buy') for x in last_60])
	n_buy = float(sum([int(x[1] == 'sell') for x in last_60]))
	amt_sell = numpy.mean([float(x[0]) for x in last_60 if x[1] == 'sell'])
	log_buy_sell_ratio = math.log(n_buy / n_sell) 
	ticker = get_ticker()['ticker'] # HISTORICAL?

	features = [log_buy_sell_ratio, float(ticker['sell']), float(ticker['buy']), float(ticker['last']), float(ticker['vol']), float(ticker['high']), float(ticker['low'])]

	assert len(features) == 7

	# 1 hour of minutes
	features += aggregated_prices(prices, 1415604000000, 60, 60)

	assert len(features) == 67

	# 1 day of hours
	features += aggregated_prices(prices, curtime - 1000 * 60 * 60, 24, 60 * 60)

	assert len(features) == 67 + 24 

	# 60 days of days
	features += aggregated_prices(prices, curtime - (1000 * 60 * 60 * 25), 60, 60 * 60 * 24)

	assert len(features) == 67 + 24 + 60  
	# average price over the last 60 minutes, last 24 hours, last 60 days
