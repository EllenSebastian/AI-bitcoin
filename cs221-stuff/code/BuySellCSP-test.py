# initialize some amt. of bitcoins you have and are willing to invest (in bitcoins to avoid value paradoxes )
# the csp decides what to do at CURRENT TIMESTAMP, not next timestamp 

import NeuralNetwork, BuySellCSP, random, pickle, pdb, dataFetcher, linecache
import numpy as np
class BuySellCSPTest: 
	# prediction is NeuralNet; perfect; or perfectRandom
	def __init__(self, nBTC, boughtAt, maxnBTC, buySellStep, actualPrices, timestep = 60, predictionMethod='NeuralNet'): 
		self.nBTC = nBTC
		self.maxnBTC = maxnBTC
		self.boughtAt = boughtAt
		self.initialBTC = nBTC 
		self.invested = 0 # NOT COUNTING the initial investment before running this ... total investment even when sold
		self.initialInvestment = self.nBTC * self.boughtAt 
		self.actualPrices = actualPrices
		self.buySellStep = buySellStep
		self.predictionMethod = predictionMethod
		self.timestep = timestep
		self.income = 0 
	# pricePrediction is for the next timestep . 
	def simulate_ts(self, ts, pricePrediction, priceData): 
		#pdb.set_trace()
		csp = BuySellCSP.BuySellCSP(pricePrediction = pricePrediction, actualPrices = priceData, nBTC = self.nBTC, boughtAt = self.boughtAt, ts = ts, btclimit = self.maxnBTC, timestep = self.timestep, nothingWeight = 3)# initialize csp 
		solution = csp.solve()['action_at_cur_ts']
		if self.nBTC < self.maxnBTC and solution == 'buy': 
			self.invested += priceData[ts]
			self.boughtAt = ((self.nBTC * self.boughtAt) + priceData[ts]) / float(self.nBTC + 1)
			self.nBTC += 1 
			if self.nBTC > self.maxnBTC: 
				print 'bought too many btc'
			print 'bought 1 bitcoin at {0}'.format(priceData[ts])
		elif solution == 'sell': # boughtAt is still the same, invested is still the same 
			self.nBTC -= 1 
			self.income += priceData[ts]
			print 'sold 1 bitcoin at {0}'.format(priceData[ts])
	# return percent profit
	def simulate(self, ts_range, priceData): 
		#pdb.set_trace()
		initialBTCWealth = priceData[ts_range[0]] * self.nBTC
		# get price predcitions
		for ts in ts_range: 
			if self.predictionMethod == 'perfectRandom': 
				pricePrediction = priceData[ts + self.timestep] + random.random()# 
			elif self.predictionMethod == 'perfect':
				pricePrediction = priceData[ts + self.timestep]
			else: 
				net = NeuralNetwork.NeuralNetwork(ts, priceData, 24, 10, 200)
				pricePredition = net.predictPrice(ts, 1, priceData) # ERROR :(
			self.simulate_ts(ts, pricePrediction, priceData)
	
		currentBtcWealth = priceData[ts_range[len(ts_range) - 1]] * self.nBTC
		profit = float(self.income + currentBtcWealth) / (self.invested + initialBTCWealth)
		print 'initialBTC: {0}, nBTC: {1}, invested: {2}, income: {3}, profit%: {4}'.format(self.initialBTC, self.nBTC, self.invested, self.income, profit)
		return {'profit':profit, 'nBTC': self.nBTC, 'invested': self.invested, 'income': self.income}
	def randomCSPSimulate(self,timestep = 60, ntimes=60*24, n=100, min_ts = None):
		profits = {} 
		self.timestep = timestep
		if min_ts is None: 
			min_ts = min(self.actualPrices.keys())
		allowedTs = [i for i in self.actualPrices.keys() if i > (min_ts + (timestep + 1) * ntimes)]
		#pdb.set_trace()
		for j in xrange(n):
			start = random.choice(allowedTs) # is actually the end, need to check the beginning					
			boughtAt = start - timestep
			while boughtAt not in self.actualPrices.keys(): 
				boughtAt -= timestep
			self.boughtAt = self.actualPrices[start - timestep]
			maxts = start - timestep
			mints = maxts - (timestep * ntimes)
			range_needed = range(mints, maxts, timestep)
			#pdb.set_trace()
			try: 
				price_subset = dataFetcher.aggregated_prices(self.actualPrices, maxts + 2 * timestep, (maxts - mints)/ timestep + 3,  timestep, 'hash')
			except Exception, e: 
				pdb.set_trace()
			#pdb.set_trace()
			prof = self.simulate(range(mints, maxts, timestep), price_subset)
			if prof is not None: 
				profits[start] = prof
			self.income = 0
			self.invested = 0
			print profits
		# xpdb.set_trace()
		return profits, np.mean([i['profit'] for i in profits.values()])


import pickle, dataFetcher
priceData = pickle.load(open('../data/bitcoin_prices.pickle'))
"""
csptest = BuySellCSPTest(nBTC = 3, boughtAt = 300, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 60, predictionMethod = 'perfect')
profits = csptest.randomCSPSimulate() # 1.0158806245527092


csptest = BuySellCSPTest(nBTC = 0, boughtAt = 0, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 60, predictionMethod = 'perfect')
profits = csptest.randomCSPSimulate() # 1.0158806245527092

csptest = BuySellCSPTest(nBTC = 0, boughtAt = 0, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 3600, predictionMethod = 'perfect')
profits = csptest.randomCSPSimulate() # 1.0166021346380312
.. now 1.00012945050636052
"""

csptest = BuySellCSPTest(nBTC = 0, boughtAt = 0, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 3600, predictionMethod = 'perfect')
profits = csptest.randomCSPSimulate(timestep = 3600, ntimes = 24, n=100, min_ts = 1387174080) # 1.0038302938953827
pdb.set_trace()

# test trying to buy


#fakePrice = {1: 20, 2: 19, 3: 18, 4: 17, 5: 16, 6: 15, 7: 14, 8:13, 9:12, 10:11, 11:10, 12:9, 13:8, 14:7, 15:6, 16:4, 17:3, 18:2, 19:1, 20:2, 21:3, 22:4}
#csptest = BuySellCSPTest(nBTC = 3, boughtAt = 20, maxnBTC = 10, buySellStep = 1, actualPrices = fakePrice, timestep = 1, predictionMethod = 'perfect')

#profits = csptest.randomCSPSimulate(1, 10, 10) # 1.0166021346380312

"""
PROBLEM: price of btc decreases after buying; do not profit e.g. is important to wait until trough to buy 
bought 1 bitcoin at 17
bought 1 bitcoin at 16
bought 1 bitcoin at 15
bought 1 bitcoin at 14
bought 1 bitcoin at 13
bought 1 bitcoin at 12
bought 1 bitcoin at 11
initialBTC: 3, nBTC: 10, invested: 98, income: 0, profit%: 0.547945205479
{15: {'profit': 0.54794520547945202, 'nBTC': 10, 'invested': 98, 'income': 0}}


# this DOES let me buy 

"""


"""
# sell some stuff
import pickle, dataFetcher
priceData = pickle.load(open('../data/bitcoin_prices.pickle'))
dataFetcher.aggregated_prices(priceData, max(priceData.keys()), (max(priceData.keys()) - min(priceData.keys())) / 60, 60, 'hash')

minPerDay = 24 * 60
maxDay = max(priceData.keys()) - 60 
minDay = maxDay - (24 * 60 * 60)
priceRange = [i for i in priceData.keys() if i >= minDay and i <= maxDay 
csptest = BuySellCSPTest(nBTC = 3, boughtAt = 300, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 60, predictionMethod = 'perfect')
csptest.simulate(priceRange, priceData)

# do not sell :(
csptest = BuySellCSPTest(nBTC =3, boughtAt = 300, 


FIRST CSP: 
start with a set amount of btc and the requirement to sell all btc within a specified time period.
The CSP chooses the best time to sell (e.g. predicted peaks in btc price) given a series of future price predictions.
This CSP found an optimal solution given perfect price predictions, but it is not very useful because
it does not take care of buying Bitcoin. It was very computationally complex due to the large number of variables
that need to be assigned (O(number of time steps)).

SECOND CSP: 
This algorithm starts with a set number of btc in the wallet (this number can be zero) and 
a time period over which to act in order to maximize profit. 
This CSP only takes past prices single future price prediction as input because we were not confident in the quality of our 
price predictions. 

We generate a new CSP on every time step. This CSP must assign only a single variable: what action to take on that specific timestep. 
It is not guaranteed to generate the optimal policies given perfect predictions, but it does take care of both buying and selling Bitocin.

The algorithm cannot be responsible for losing money using Bitcoin because it will never sell bitcoins at a loss. 
The only way one can lose money using this algorithm is if one buys Bitcoin at a high price above which the price never
rises during the provided time period. 

This CSP usually produces a small amount of profit, about 1.5% over 1440 minutes (1 day) or 1440 hours. 
We observed the issue that the algorithm often lost money due to selling too early, e.g. selling before reaching a peak or buying before reaching a trough. 
This meant that the profit was bounded even with perfect price predictions.
For example, in a controlled test case where the price decreased from $20 to $10 by $1 each minute, the algorithm bought 1 bitcoin
each minute after observing the trend of decreasing prices. As a consequence, our profit was -46%, since we invested more than the final value of our bitcoin assets. 



THIRD CSP:
The main problem with the second CSP was that it did not detect peaks and troughs in Bitcoin data. We therefore experimented
with incorporating price predictions further into the future. If we could see that we had not yet reached a peak, we would not sell yet, and 
if we coud see that we had not yet reached a trough, we would not buy yet. THis CSP would be much more effective than the second CSP for the case
of good price predictions. However, it should perform poorly with bad price predictions, as we will not be able to detect peaks or troughs. 

We found that the rule-based algorithm generally acts optimally. For example, in an instance where Bitcoin price never rises above
the price Bitcoins were bought at, we do not sell any bItocins. 


US bonds: 2% per year in 2012, may even have negative in future: http://observationsandnotes.blogspot.com/2010/11/100-years-of-bond-interest-rate-history.html
US bonds: 1.5 to 5% per year in the last decade
US stocks: 9% yield in the last deceade http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html

we must do better than them
"""


