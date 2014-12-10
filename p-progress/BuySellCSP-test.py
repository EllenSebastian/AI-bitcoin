# initialize some amt. of bitcoins you have and are willing to invest (in bitcoins to avoid value paradoxes )
# the csp decides what to do at CURRENT TIMESTAMP, not next timestamp 

import NeuralNetwork, BuySellCSP, random, pickle, pdb, dataFetcher, linecache
class BuySellCSPTest: 
	# prediction is NeuralNet; perfect; or perfectRandom
	def __init__(self, nBTC, boughtAt, maxnBTC, buySellStep, actualPrices, timestep = 60, predictionMethod='NeuralNet'): 
		self.nBTC = nBTC
		self.maxnBTC = maxnBTC
		self.boughtAt = boughtAt
		self.initialBTC = nBTC 
		self.invested = 0 # NOT COUNTING the initial investment before running this ... total investment even when sold
		self.initialInvestment = self.nBTC * self.boughtAt 
		self.actualPrices  = dataFetcher.aggregated_prices(actualPrices, max(actualPrices.keys()), (max(actualPrices.keys()) - min(actualPrices.keys())) / timestep,  timestep, 'hash')
		pdb.set_trace()
		self.buySellStep = buySellStep
		self.predictionMethod = predictionMethod
		self.timestep = timestep
		self.income = 0 
	# pricePrediction is for the next timestep . 
	def simulate_ts(self, ts, pricePrediction): 
		csp = BuySellCSP.BuySellCSP(pricePrediction = pricePrediction, actualPrices =  self.actualPrices, nBTC = self.nBTC, boughtAt = self.boughtAt, ts = ts, btclimit = self.maxnBTC, timestep = self.timestep, nothingWeight = 3)# initialize csp 
		solution = csp.solve()['action_at_cur_ts']
		if self.nBTC < self.maxnBTC and solution == 'buy': 
			self.invested += self.actualPrices[ts]
			self.boughtAt = ((self.nBTC * self.boughtAt) + self.actualPrices[ts]) / float(self.nBTC + 1)
			self.nBTC += 1 
			if self.nBTC > self.maxnBTC: 
				print 'bought too many btc'
			print 'bought 1 bitcoin at {0}'.format(self.actualPrices[ts])
		elif solution == 'sell': # boughtAt is still the same, invested is still the same 
			self.nBTC -= 1 
			self.income += self.actualPrices[ts]
			print 'sold 1 bitcoin at {0}'.format(self.actualPrices[ts])
	# return percent profit
	def simulate(self, ts_range, priceData): 
		initialBTCWealth = priceData[ts_range[0]] * self.nBTC
		# get price predcitions
		for ts in ts_range: 
			try: 
				if self.predictionMethod == 'perfectRandom': 
					pricePrediction = self.actualPrices[ts + self.timestep] + random.random()# 
				elif self.predictionMethod == 'perfect':
					pricePrediction = self.actualPrices[ts + self.timestep]
				else: 
					net = NeuralNetwork.NeuralNetwork(ts, priceData, 24, 10, 200)
					pricePredition = net.predictPrice(ts, 1) # ERROR :(
				self.simulate_ts(ts, pricePrediction)
			except Exception, e: 
				return None 	
		currentBtcWealth = self.actualPrices[ts_range[len(ts_range) - 1]] * self.nBTC
		profit = float(self.income + currentBtcWealth) / (self.invested + initialBTCWealth)
		print 'initialBTC: {0}, nBTC: {1}, invested: {2}, income: {3}, profit%: {4}'.format(self.initialBTC, self.nBTC, self.invested, self.income, profit)
		return profit
	def randomCSPSimulate(self,timestep = 60, ntimes=60*24, n=100):
		profits = {} 
		for j in xrange(n):
			start = random.choice(priceData.keys())
			maxts = start - timestep
			mints = maxts - (timestep * ntimes)
			priceRange = sorted([i for i in priceData.keys() if i >= mints and i <= maxts])
			prof = self.simulate(priceRange, priceData)
			if prof is not None: 
				profits[start] = prof
			self.income = 0
			self.invested = 0
			print profits
		return profits 


csptest = BuySellCSPTest(nBTC = 3, boughtAt = 300, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 60, predictionMethod = 'perfect')
profits = csptest.randomCSPSimulate() # 1.0158806245527092


csptest = BuySellCSPTest(nBTC = 0, boughtAt = 0, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 60, predictionMethod = 'perfect')
profits = csptest.randomCSPSimulate() # 1.0158806245527092

csptest = BuySellCSPTest(nBTC = 0, boughtAt = 0, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 3600, predictionMethod = 'perfect')
profits = csptest.randomCSPSimulate() # 1.0166021346380312


""""
# sell some stuff
import pickle, dataFetcher
priceData = pickle.load(open('../data/bitcoin_prices.pickle'))
dataFetcher.aggregated_prices(priceData, max(priceData.keys()), (max(priceData.keys()) - min(priceData.keys())) / 60, 60, 'hash')

minPerDay = 24 * 60
maxDay = max(priceData.keys()) - 60 
minDay = maxDay - (24 * 60 * 60)
priceRange = [i for i in priceData.keys() if i >= minDay and i <= maxDay ]
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





"""


