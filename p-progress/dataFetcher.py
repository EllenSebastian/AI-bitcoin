import urllib, json, datetime
n_transactions_wanted = 60 # number of transactions to keep per minute
def read_data(file): 
	out = {}
	for line in open(file): 
		line_split = line.strip().split(',')
		out[time_to_date(time.strptime(line.strip().split(',')[0], time_format) )] = float(line.strip().split(',')[1])
	return out 


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

def make_feature_vector_from_file(data, start_date, end_date):
    out = []
    cur_date = copy.deepcopy(start_date)
    while cur_date <= end_date: 
        if cur_date in data: 
            out.append(data[cur_date])
        else: 
            out.append(None)
        cur_date += timedelta(days=1)
    return out 

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

# must call pickle.load('bitcoin_prices.pickle') to get the prices and pass them in. 
def aggregated_prices(prices, end_timestamp, n_aggregates = 100, aggregation= 60): 
	aggregation *= 1000
	start_timestamp = end_timestamp - n_aggregates * aggregation
	sorted_timestamps = sorted([x for x in prices.keys() if x >= start_timestamp and x <= end_timestamp])
	out = []
	for i in range(n_aggregates): 
		matches = [prices[x] for x in sorted_timestamps if x >= start_timestamp and x < (start_timestamp + aggregation)]
		out.append(numpy.mean(matches))
		start_timestamp += aggregation
	return out 

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

