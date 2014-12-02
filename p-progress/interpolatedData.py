import urllib, json, datetime, time, pickle, schedule, pdb
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
        print f 
    	out[f] = {} 
        data = read_data('../data/' + f, True)
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
    	pickle.dump(out[f], open('../data/' + f.split('.')[0] + str(step) + '.pickle','w'))

interpolatedData()