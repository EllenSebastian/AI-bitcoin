"""
CSP

"""


# pricePredictions should be like {ts1: {ts2: xx, ts3:xx}, ts2: {ts3:xx}}
# e.g. {knownTs: {unknownTs: unknownPrice}}
priceData = pickle.load(open('../data/bitcoin_prices.pickle'))

def trading_test(timestep = 60 * 60, n_timesteps = 24, nBtc = 3, boughtAt = 380, start_ts = 1411085400): 
	pricePredictions = {}
	# predict start_ts, use the previous ts to get a known ts

	for i in range(n_timesteps):
		cur_start_ts = start_ts + i * timestep 
		pricePredictions[cur_start_ts] = net.predictPrice(cur_start_ts, n_timesteps - i)[1]
		prevPrice = priceData[cur_start_ts - timestep]
		bCSP = BitcoinCSP.BitcoinCSP(pricePredictions[cur_start_ts], nBtc, boughtAt, prevPrice, timestep)
		soln = bCSP.solve()

def maxProfit(start_ts, timestep, n_timesteps, nBtc, boughtAt):
	maxProfit = 0 
	for i in range(n_timesteps):
		ts = start_ts + i * timestep 


net = NeuralNetwork.NeuralNetwork(endTimeStamp = 1411175400)

pred = net.predictPrice(1411175400-3600)




	def simulate(self):
		cur_start_ts = self.start_ts
		while cur_start_ts <= self.end_ts:  # cur_start_ts is a prediction
			print 'starting from {0}'.format(cur_start_ts)
			self.makeCSP(cur_start_ts)
			asst = self.solve()
			if asst[cur_start_ts + self.timestep] > 0: 
				self.nBTC -= asst[cur_start_ts + self.timestep]
				self.profit += asst[cur_start_ts + self.timestep] * self.pricePredictions[cur_start_ts][cur_start_ts + self.timestep]
				print 'sell {0} bitcoin at {1} for {2}'.format(asst[cur_start_ts], cur_start_ts, self.pricePredictions[cur_start_ts][cur_start_ts + self.timestep])
			cur_start_ts += self.timestep
		return profit
