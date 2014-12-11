import urllib, json, datetime, time, pickle, schedule, pdb, sys, linecache
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
time_format = '%d/%m/%Y %H:%M:%S'

n_transactions_wanted = 60 # number of transactions to keep per minute
def read_data(file, timestamp = False): 
	out = {}
	for line in open(file): 
		line_split = line.strip().split(',')
		if timestamp: 
			out[int(time.mktime(time.strptime(line.strip().split(',')[0],time_format)))] = float(line.strip().split(',')[1])
		else: 
			out[time_to_date(time.strptime(line.strip().split(',')[0], time_format) )] = float(line.strip().split(',')[1])
	return out 


def storeDataAsPickle():
	return 

# start_ts is the unix timestamp to start at
# step is how often to return a point in seconds
def interpolatedData(start_ts = None, end_ts = None, step = 60, files = ['avg-confirmation-time.txt', 'estimated-transaction-volume.txt', 'my-wallet-transaction-volume.txt', 'total-bitcoins.txt', 
                     'bitcoin-days-destroyed-cumulative.txt','hash-rate.txt', 'n-orphaned-blocks.txt','trade-volume.txt', 'bitcoin-days-destroyed.txt','market-cap.txt', 
                     'n-transactions-excluding-popular.txt','transaction-fees.txt', 'blocks-size.txt','n-transactions-per-block.txt', 'tx-trade-ratio.txt', 
                     'cost-per-transaction.txt','miners-revenue.txt', 'n-transactions.txt', 'difficulty.txt','my-wallet-n-tx.txt', 'n-unique-addresses.txt', 
                     'estimated-transaction-volume-usd.txt', 'my-wallet-n-users.txt', 'output-volume.txt']): 
    out = {}
    for f in files: 
    	out[f] = {} 
        data = read_data('data/' + f, True)
        x = [] 
        y = []
        sort_indices = sorted(enumerate(data.keys()), key=lambda x: x[1])
        for k in sort_indices: 
        	x.append(k[1])
        	y.append(data[k[1]])
        #pdb.set_trace()
        interpolator = interp1d(x, y, kind='cubic')
        xnew = []
        ynew = []
        if start_ts is None: cur_start_ts = int(min(x))
    	else: cur_start_ts = start_ts
    	if end_ts is None: cur_end_ts = int(max(x))
    	else: cur_end_ts = end_ts
    	new_y = interp1d(x, y, kind='cubic')(range(cur_start_ts, cur_end_ts, step))
    	out[f] = dict(zip(range(cur_start_ts, cur_end_ts, step), new_y))
    	#pdb.set_trace()
    	#plt.plot(x,y,'bo', range(cur_start_ts, cur_end_ts, step),new_y, 'r--')
    	#plt.show()
    	pickle.dump(out[f], open(f.split('.')[0] + str(step) + '.pickle','w'))
    return out 


def time_to_date(t): 
	dt = datetime.datetime.fromtimestamp(time.mktime(t))
	return datetime.date(dt.year, dt.month, dt.day)


def day_to_price():
	out = {}
	for line in open('data/market-price.txt'): 
		line_split = line.strip().split(',')
		t = datetime.date(datetime.fromtimestamp(time.mktime(time.strptime(line_split[0], time_format))))
		f = float(line_split[1])
		out[t] = f
	return out

def get_json(url): 
	response = urllib.urlopen(url)
	return json.loads(response.read())

def read_data(file): 
	out = {}
	for line in open(file): 
		line_split = line.strip().split(',')
		out[time_to_date(time.strptime(line.strip().split(',')[0], time_format) )] = float(line.strip().split(',')[1])
	return out 


def make_feature_vector_from_file(data, start_date, end_date):
    out = []
    cur_date = copy.deepcopy(start_date)
    while cur_date <= end_date: 
        if cur_date in data: 
            out.append(data[cur_date])
        else: 
            out.append(None)
        cur_date += datetime.timedelta(days=1)
    return impute_vector(out)

def impute_vector(vec): 
    imputed_assignments = {}
    for i in range(len(vec)): 
    	if vec[i] is None: 
    		# find the previous data point
    		prev, next = None, None
    		for j in range(i-1,0, -1): 
    			if vec[j] is not None: 
    				prev = vec[j]
    				break
    		for j in range(i+1,len(vec)): 
    			if vec[j] is not None: 
    				next = vec[j]
    		if next is not None and prev is not None: 
    			imputed_assignments[i] = np.mean([next, prev])
    		elif next is not None: 
    			imputed_assignments[i] = next
    		elif prev is not None: 
    			imputed_assignments[i] = prev
    for i in imputed_assignments.keys(): 
    	vec[i] = imputed_assignments[i]
    return vec

# e.g. {u'date': 1415750105, u'tid': 762497, u'amount': u'0.09', u'type': u'sell', u'price': u'366.43'}
def get_last_60_transactions(): 
	return get_json('https://www.okcoin.com/api/trades.do?ok=1')

def get_ticker(): 
	return get_json('https://www.okcoin.com/api/ticker.do?ok=1')



# URL:http://api.coindesk.com/charts/data?data=close&startdate=2012-11-10&enddate=2012-11-10&exchanges=bpi&dev=1&index=USD&callback=cb
# endTime is the Unix time stamp to count backwards from for aggregation. 
# start_date and end_date are in GMT!! (probably)
def read_granular_transactions(start_date=datetime.datetime(2010, 12, 01), end_date=datetime.datetime(2014, 11, 11)): 
	step = datetime.timedelta(days=1)
	dates = []
	while start_date < end_date:
	    dates.append(start_date.strftime('%Y-%m-%d'))
	    start_date += step
	pdb.set_trace()
	prices = {}
	for date_i in range(len(dates) - 1): 
		if date_i < 10: 
			pdb.set_trace()
		print date_i
		url = "http://api.coindesk.com/charts/data?data=close&startdate=" + dates[date_i] + '&enddate=' + dates[date_i + 1] + '&exchanges=bpi&dev=1&index=USD&callback=cb'
		response = urllib.urlopen(url)
		dat = response.read()
		for line in json.loads(dat[3:len(dat) - 2])['bpi']:
			prices[line[0] / 1000] = line[1] # convert to seconds 
	return prices 

# TODO update the data
def impute_hash(out, aggregation): 
	try: 
		for k in sorted(out.keys()): 
			if out[k] is None: 
				prev, next = k,k
				while prev in out.keys() and out[prev] is None:
					prev -= aggregation
				while next in out.keys() and out[next] is None:
					next += aggregation
				if (next in out.keys() and out[next] is not None) and (prev in out.keys() and out[prev] is not None):
					out[k] = np.mean([out[next], out[prev]])
				elif next in out.keys() and out[next] is not None: 
					out[k] = out[next]
				else: 
					out[k] = out[prev]
	except Exception, e: 
		pdb.set_trace()
	return out 


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)


# must call pickle.load('bitcoin_prices.pickle') to get the prices and pass them in. 
def aggregated_prices(prices, end_timestamp, n_aggregates = 100, aggregation= 60, returnT="arr"): 
	"""
	------------------------------------------------------------------------------
	prices - dict of unix timestamps to price
	end_timestamp - unix timestamp at which the procurement ends 
	n-aggregates - number of (timestamp : price) elements returned
	aggregation - number of seconds to aggregate over 
	return list of prices in chronological order ending at end_timestamp
	------------------------------------------------------------------------------
	"""
	try: 
		start_timestamp = end_timestamp - n_aggregates * aggregation
		sorted_timestamps = sorted([x for x in prices.keys() if x >= start_timestamp and x <= end_timestamp])
		if returnT == "arr": out = []
		else: out = {}
		cur_ts = 0
		for i in range(n_aggregates):
			if i % 10000 == 0: 
				print i  
			matches = []
			while sorted_timestamps[cur_ts] < (start_timestamp + aggregation): 
				cur_ts += 1
				matches.append(prices[sorted_timestamps[cur_ts]])
			if len(matches) == 0: 
				meann = None
			else:
				meann = np.mean(matches)
			if returnT == "arr":
				out.append(meann)
			else:
				out[start_timestamp] = meann
			start_timestamp += aggregation
		if returnT == 'arr': 
			return impute_vector(out)
		else:
			return impute_hash(out, aggregation)
	except Exception, e:
		PrintException()
		pdb.set_trace() 



def aggregated_data(data, end_timestamp, n_aggregates = 100, aggregation = 60, whichData="prices"):
	"""
	------------------------------------------------------------------------------
	data - dict of unix timestamps to price
	end_timestamp - unix timestamp at which the procurement ends 
	n-aggregates - number of (timestamp : price) elements returned
	aggregation - number of seconds to aggregate over 
	return list of data in chronological order ending at end_timestamp
	------------------------------------------------------------------------------
	"""
	try:
		start_timestamp = end_timestamp - n_aggregates * aggregation
		time_range = (start_timestamp, end_timestamp)
		sorted_timestamps = sorted([x for x in data.keys() if x >= start_timestamp and x <= end_timestamp])
		mappedData = {}
		listData = []
		mappedListData = []
		cur_ts = 0
		for i in range(n_aggregates):
			matches = []
			while sorted_timestamps[cur_ts] < (start_timestamp + aggregation):
				cur_ts += 1

				# different kinds of aggregation maybe needed for different kinds of data
				matches.append(data[sorted_timestamps[cur_ts]])
				mean = np.mean(matches)
			listData.append(mean)
			mappedData[start_timestamp] = mean
			mappedListData.append([start_timestamp, mean])
			start_timestamp += aggregation
		return listData, mappedData, mappedListData, time_range
	except Exception, e:
		pdb.set_trace()

def print_prices(prices): 
	f = open('prices.csv','w')
	for key in sorted(prices.keys()): 
		f.write('{0},{1}'.format(key, prices[key]))
	f.close()

# 60 transactions 
def record_transactions_per_minute(): 
	# {minute_modded_timestamp-> buy/sell ratio, avg amount, avg sell amount, avg buy amount}
	out = {}	
	since = 26000
	while True: 
		print since, len(out.keys())
		trades = get_json('https://www.okcoin.com/api/v1/trades.do?since={0}'.format(since))
		for trade in trades: 
			key = trade['date'] - (trade['date'] % 60)
			if key not in out:
				out[key] = []
			if len(out[key]) < 60: 
				out[key].append((trade['amount'], trade['type']))
		if trades == []: 
			break
		since = trades[len(trades) - 1]['tid']
	return out 


def get_order_book():
    with open('order_book.pickle', 'a+b') as f:
        order_book = {}
        timestamp = int(time.time())
        order_book[timestamp] = get_json('https://www.okcoin.com/api/depth.do?ok=1')
        print 'write', timestamp
        pickle.dump(order_book, f)

def continue_getting_order_book():
    schedule.every(30).seconds.do(get_order_book)
    while 1:
        schedule.run_pending()
        time.sleep(1)
