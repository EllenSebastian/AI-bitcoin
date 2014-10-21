import time, datetime, copy
from datetime import date
from datetime import timedelta
from datetime import datetime
time_format = '%d/%m/%Y %H:%M:%S'

def time_to_date(t): 
	return datetime.date(datetime.fromtimestamp(time.mktime(t)))
# return 2d array of data in file 
# first index of each row is a Time, second index is a Float
def read_data(file): 
	out = {}
	for line in open(file): 
		line_split = line.strip().split(',')
		out[time_to_date(time.strptime(line.strip().split(',')[0], time_format) )] = float(line.strip().split(',')[1])
	return out 

# return the 
def remove_nones(all_features, Y):
	Y_clean = []
	all_features_clean = []
	for i in xrange(len(all_features)):
	    cur = []
	    if None not in all_features[i]: 
	        for j in all_features[i]:
	            cur.append(int(j))
	        all_features_clean.append(cur)
	        Y_clean.append(Y[i])
	return all_features_clean, Y_clean


def date_to_time(date): 
    return time.strptime('{0}/{1}/{2} 00:00:00'.format(date.day, date.month, date.year), '%d/%m/%Y %H:%M:%S')


# aggregation can be 'average' or 'sum'
# agg_length is an integer of days
# return an array of values from start_ts

def day_to_price():
	out = {}
	for line in open('data/market-price.txt'): 
		line_split = line.strip().split(',')
		t = datetime.date(datetime.fromtimestamp(time.mktime(time.strptime(line_split[0], time_format))))
		f = float(line_split[1])
		out[t] = f
	return out

def make_feature_vector_from_file(data, start_ts, end_ts, aggregation = None, agg_length = None):
	i = len(data) - 1 
	out = []
	while(True): 
		if i < 0: 
			break
		if data[i][0] > end_ts: 
			i -= 1
			continue
		if data[i][0] < start_ts: 
			break
		if aggregation == None: 
			out.append(data[i][1])
			i -= 1 
		elif aggregation == 'average' and i > agg_length: 
			sum = 0 
			for j in range(0, agg_length): 
				sum += data[i][1]
				i -= 1
			out.append(sum / float(agg_length))
		elif aggregation == 'sum' and i > agg_length: 
			sum = 0 
			for j in range(0, agg_length): 
				sum += data[i][1]
				i -= 1
			out.append(sum)
		else: 
			i -= 1 
	return out 



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
