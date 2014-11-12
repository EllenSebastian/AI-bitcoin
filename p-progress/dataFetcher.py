import urllib, json, datetime
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
			prices[line[0]] = line[1]
	return prices 

def print_prices(prices): 
	f = open('prices.csv','w')
	for key in sorted(prices.keys()): 
		f.write('{0},{1}'.format(key, prices[key]))
	f.close()
