# this CSP decides whether to buy, sell, or do nothing at a single time step. 
# we are looking for the assignment of a single variable (-1, 0, 1)
# it depends primarily on a single price prediction in the future
import pickle, dataFetcher, BitcoinCSP, CSP, math, CSPsolver, pdb



# buy on increases and troughs
# sell on decreases and peaks
class BuySellCSP: 
	# pricePrediction: prediction for btc price at the next ts 
	# actualPrices; hash from ts to btc price
	# boughtAt: price or average price the btc were bought at 
	# ts = current (not future) ts
	def __init__(self, pricePrediction, actualPrices, nBTC, boughtAt, ts, btclimit=20, timestep=60, nothingWeight = 100):
		self.pricePrediction = pricePrediction
		self.nothingWeight = nothingWeight
		self.nBTC = nBTC
		self.csp = CSP.CSP()
		self.boughtAt = boughtAt
		self.actualPrices = actualPrices
		self.timestep = timestep
		self.csp.add_variable('action_at_cur_ts', ['sell', 'nothing', 'buy']) # may change to better domain later
		self.btclimit = btclimit
		def sell_higher(val): 
			if self.boughtAt < self.actualPrices[ts] and val == 'sell': 
				return 1 # ok to sell at a high price
			elif self.boughtAt < self.actualPrices[ts] and val == 'buy':
				return 0.5 # technically ok to buy at a higher price than you did before
			elif self.boughtAt > self.actualPrices[ts] and val == 'sell': 
				# trying to sell at a low price
				return 0 # do not sell at a low price
			elif self.boughtAt > self.actualPrices[ts] and val == 'buy': 
				return 1 # good to buy at a low price  
			elif val == 'nothing':
				return 1 # always ok to do nothing
			return 0 # don't sell at a lower price
		# here, check for INCREASES of prediction over current price --> buy
		def buy_at_trough(val): 
			if val == 'buy' and self.nBTC >= self.btclimit: 
				return 0
			print 'pricePrediction: {0}, actual: {1}'.format(pricePrediction, self.actualPrices[ts])
			if pricePrediction > self.actualPrices[ts] and val == 'buy': # maybe a trough: buy when price will increase
				# scale by the number of price decreases before 
				ts_prev = ts
				nDecrease = 0 
				# checking for decreases. if it was a straight increase, better to buy. if it's a trough, also buy. 
				for i in xrange(1,20):
					ts_prev = ts - (i * self.timestep)
					if self.actualPrices[ts_prev] < self.actualPrices[ts_prev 0 self.timestep]:
						nDecrease += 1
					# the longer the increase lasted, the more likely it is a trough
					# the longer the decrease lasted, the more likely it is a trough
				print 'returning nDecrease = {0}'.format(nDecrease)
				return float(nDecrease) / 10# return at least 2: better to buy on an increase, even better if it's a trough.
			return 1 # i don't care 
		# here, check for DECREASES of prediction over current price --> sell
		def sell_at_peak(val): 
			if val == 'sell' and nBTC <= 0: 
				return 0 
			if val == 'sell' and self.pricePrediction >= self.boughtAt:  # never sell at a lower price than you bought
				return 0
			if pricePrediction < self.actualPrices[ts] and val == 'sell': # this is not a peak
				# scale by the number of price decreases before 
				nIncrease = 0 
				for i in xrange(1,21):
					ts_prev = ts - (i * self.timestep)
					if self.actualPrices[ts_prev] > self.actualPrices[ts_prev - self.timestep]:
						nIncrease += 1
					# the longer the increase lasted, the more likely it is a trough
				#print 'returning nIncrease = {0}'.format(nIncrease)
				return float(nIncrease) / 10.0 # better to buy on a decrease, even better if it's a peak.
			return 1 # i don't care 
		def weighting(val): 
			if val == 'nothing': 
				return self.nothingWeight 
			else: 
				return 1 
			# or if the ratio between sellweight and buyweight is below a certain threshold, then do nothing
			# or, do nothing only if that is the only thing i CAN"T do (can't buy or sell)
		self.csp.add_unary_potential('action_at_cur_ts', sell_higher)
		self.csp.add_unary_potential('action_at_cur_ts', buy_at_trough)
		self.csp.add_unary_potential('action_at_cur_ts', sell_at_peak)
	def solve(self):
		search = CSPsolver.BacktrackingSearch()
		search.solve(self.csp)
		#print search.optimalAssignment
		return search.optimalAssignment


#bCSP = BuySellCSP(pricePrediction= predictedPrice, actualPrices=prices24, nBTC=4, boughtAt=260, ts=max(prices24.keys()), timestep=60, nothingWeight = 100)  
# bCSP.solve()
# YAY BASELINEs

# WILL SEOMTIMES SELL NOW 


"""
prices = {1413237600: 400, 1413252000: 450}
import pickle, dataFetcher, BuySellCSP
priceData = pickle.load(open('../data/bitcoin_prices.pickle'))
prices24 = dataFetcher.aggregated_prices(priceData, end_timestamp = 1413590400, n_aggregates = 24, aggregation = 60,returnT = 'hash' )
#prices24 = {1413504000: 378.54483333333337, 1413532800: 376.85616666666664, 1413568800: 376.90699999999998, 1413597600: 376.178, 1413626400: 379.63516666666663, 1413655200: 380.37383333333332, 1413511200: 378.66600000000005, 1413547200: 380.77499999999998, 1413576000: 376.28866666666664, 1413604800: 375.45466666666675, 1413633600: 380.88550000000004, 1413662400: 380.91816666666665, 1413518400: 379.62283333333329, 1413540000: 377.80383333333333, 1413554400: 380.02983333333327, 1413583200: 373.71899999999994, 1413612000: 375.14399999999995, 1413640800: 379.79616666666664, 1413669600: 381.14983333333328, 1413525600: 377.12666666666667, 1413561600: 378.70796610169492, 1413590400: 374.9516666666666, 1413619200: 375.86149999999992, 1413648000: 379.50849999999997}
predictedPrice = 260
bCSP = BuySellCSP(pricePrediction= predictedPrice, actualPrices=prices24, nBTC=4, boughtAt=260, ts=max(prices24.keys()) + 60, timestep=60, nothingWeight = 3)  
bCSP.solve()

"""