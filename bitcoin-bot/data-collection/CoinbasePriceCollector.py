"""
This class collects prices from the Coinbase API and outputs them to files. 
Each file is a csv where the first column is the timestamp, second column is the buy price, and the third column is the sell price.
An effort is made to have each time stamp be divisible by the interval (e.g. 1418716920 for interval=60), but this is not guaranteed to be the case.
"""

import urllib2, json, threading, time

class priceCollector: 
	"""
	frequency: how often in seconds to output a price.
	outfile: prefix of files to print to. files will be of form "outfileX.csv".
	outfileLength: maximum number of lines in an outfile.
	"""
	def __init__(self, frequency=60, outfile='btcPrices', outfileLength = 10000):
		self.frequency = frequency
		self.outfile = outfile
		self.times_run = 0
		self.outfileLength = outfileLength
		self.out = open(self.outfile + '0.csv','w')

	def priceFromURL(self, url):
		response = urllib2.urlopen(url)
		return float(json.loads(response.read())['subtotal']['amount'])

	def record_price(self):
		runtime = int(round(time.time()))
		print 'running at {0}'.format(time.time())
		if self.times_run % self.outfileLength == 0: 
			newfilename = self.outfile + str(int(time.time())) + '.csv'
			self.out.close()
			print 'opening new file: ' + newfilename
			self.out = open(newfilename, 'w')
		buyPrice = self.priceFromURL('https://api.coinbase.com/v1/prices/buy?qty=1')
		sellPrice = self.priceFromURL('https://api.coinbase.com/v1/prices/sell?qty=1')
		self.out.write('{0},{1},{2}\n'.format(runtime, buyPrice, sellPrice))
		self.out.flush()
		self.times_run += 1 

	def do_every (self, interval, worker_func, iterations = 0):

	  if iterations != 1:
	    threading.Timer (
	      interval,
	      self.do_every, [interval, worker_func, 0 if iterations == 0 else iterations-1]
	    ).start ();
	  worker_func ();

	""" 
	collect prices ntimes times.
	will run forever if ntimes is not specified.
	"""
	def run(self, ntimes=None):
		if ntimes is None: 
			ntimes = float('inf')
		while time.time() % self.frequency > 0.1: 
			time.sleep(0.0001)
		print 'starting at: {0}'.format(time.time())
		self.do_every(self.frequency, self.record_price, ntimes) 
