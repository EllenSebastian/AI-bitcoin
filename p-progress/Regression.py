# extract features for regression
import numpy, pickle, random, pdb, copy, pickle, math, random
import PricePredictor
N_EXAMPLES = 1000
DAYS_FOR_FEATURES = 30 # number of days to look in the past for non-price features. 
dt = 60 
pct_change = False # use percent change instead of raw?
use_other_features = False # use non-price inputs?

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
        n_not_found = 0 
        for j in range(0, 60): 
            if sorted_timestamps[i] - j * 60 not in prices: 
                n_not_found += 1 
        if n_not_found == 0: 
            possible_train.append(sorted_timestamps[i])
# end up with 98707 places to choose

train_examples = random.sample(possible_train, N_EXAMPLES)

all_features = []
all_Y = []

# convert [1,1.5,2] to [0.5, 0.25]
def convert_to_pct_change(vec): 
    out = []
    for i in xrange(1, len(vec)):
        out.append((vec[i] - vec[i-1]) / float(vec[i]))
    return out 

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
    # 1 hour of minutes
    minPrices = []
    for i in range(0, 60): 
        minPrices += [prices[train_ts - i*60]]
    if pct_change: 
        features += convert_to_pct_change(minPrices)
        features += convert_to_pct_change(aggregated_prices(prices, train_ts - (60 * 60), 24, 60 * 60))
        features += convert_to_pct_change(aggregated_prices(prices, train_ts - (60 * 60 * 25), 60, 60 * 60 * 24))
    else: 
        features += minPrices
        features += aggregated_prices(prices, train_ts - (60 * 60), 24, 60 * 60)
        features += aggregated_prices(prices, train_ts - (60 * 60 * 25), 60, 60 * 60 * 24)
    # 60 days of days
    features += aggregated_prices(prices, train_ts - (60 * 60 * 25), 60, 60 * 60 * 24)
    start_datetime = datetime.datetime.fromtimestamp(train_ts)
    end_day = datetime.date(start_datetime.year, start_datetime.month, start_datetime.day)
    start_day = end_day - datetime.timedelta(days=DAYS_FOR_FEATURES)
    if use_other_features: 
    	for file in non_price_inputs:
    	    cur_features = make_feature_vector_from_file(data[file], start_day, end_day) 
    	    features += cur_features 
    return features	

all_Y = []
all_features = []
i = 0
for train_ts in train_examples:
    print i 
    i += 1
    try:
        features = features_for_ts(train_ts)
    except Exception:
        features = [None]
    while None in features: 
        while train_ts in train_examples: 
            train_ts = random.choice(possible_train)
        try:
            features = features_for_ts(train_ts)
        except Exception:
            features = [None]
    all_features.append(features)
    all_Y.append((prices[train_ts + dt] - prices[train_ts]) / float(prices[train_ts]))
    # average price over the last 60 minutes, last 24 hours, last 60 days
    if train_ts == train_examples[0]: 
        pdb.set_trace()
    if len(all_Y) >= 1000: 
        break 

#gp = PricePredictor.PricePredictor(np.array(all_features), np.array(all_Y), 'gp')
#err, predictions = gp.crossValidation(10)
#print err, predictions


linear_model_ = PricePredictor.PricePredictor(all_features, all_Y, 'linear')
linear_err, linear_predictions= linear_model_.crossValidation(10)
# no outside variables: True pos 247 True neg 234 False pos 275 false_neg 244
# outside variables:    True pos 256 True neg 250 False pos 249 false_neg 245

# has very bad high predictions: 113740%

bayesian_model = PricePredictor.PricePredictor(all_features, all_Y, 'BayesianRidge')
bayes_err, bayes_predictions= bayesian_model.crossValidation(10)
# no outside variables: True pos 241 True neg 261 False pos 244 false_neg 254
# outside variables: True pos 240 True neg 259 False pos 243 false_neg 258


ridge_model = PricePredictor.PricePredictor(all_features, all_Y, 'ridge')
ridge_err, ridge_predictions= ridge_model.crossValidation(10)
# outside variables: True pos 239 True neg 206 False pos 231 false_neg 208
# no outside variables: True pos 239 True neg 257 False pos 245 false_neg 259


logistic_model = PricePredictor.PricePredictor(all_features, all_Y, 'logistic') # does not finish
err, predictions= pp_linear.crossValidation(10)
# True pos 228 True neg 271 False pos 233 false_neg 268

perceptron_model = PricePredictor.PricePredictor(all_features, all_Y, 'perceptron')
perceprton_err, perceptron_predictions= pp_linear.crossValidation(10)


# predictions and all_Y are both percent price change arrays, 
# where the index in predictions corresponds to the index in all_Y.
def plot_predictions(predictions, true_Y):
	pred = []
	for i in range(0,len(predictions)):
		pred.append(predictions[i])
	import matplotlib.pyplot as plt
	plt.scatter(true_Y, pred)
	plt.xlabel('Actual Delta P values')
	plt.ylabel('Predicted Delta P values')
	plt.title('Predicted vs actual Delta P values for Bayesian Ridge Regression')
	plt.show()
