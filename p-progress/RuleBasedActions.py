# sell at an observed trough (some decreases in the past and a predicted increase)
# sell whenever you see a price higher than what you bought at 
import NeuralNetwork, BuySellCSP, random, pickle, pdb, dataFetcher, linecache
import numpy as np
class RuleBasedActionPicker: 
    def __init__(self, nBTC, boughtAt, maxnBTC, buySellStep, actualPrices, timestep = 60, predictionMethod='NeuralNet', minProfit = 0.5):
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
        self.minProfit = minProfit # min profit to sell
        self.threshold = 0
        self.totalCash = 0

    def total_wealth(self, price):
        return self.totalCash + price * self.nBTC

    def simulate_ts(self, ts, pricePrediction, priceData): 
        if (priceData[ts] - pricePrediction) > self.threshold:
            # sell all
            #print 'sell', self.nBTC, 'bitcoins at p=', priceData[ts], 'total_wealth=', self.totalCash + priceData[ts] * self.nBTC
            #self.totalCash += priceData[ts] * self.nBTC
            #self.nBTC = 0
            # sell one
            if self.nBTC > 0:
                print 'sell 1 bitcoin at p=', priceData[ts], 'nBTC=', self.nBTC, 'total_wealth=', self.total_wealth(priceData[ts])
                self.totalCash += priceData[ts]
                self.nBTC -= 1
        elif (pricePrediction - priceData[ts]) > self.threshold:
            if self.totalCash >= priceData[ts]:
                print 'buy 1 bitcoin at p=', priceData[ts], 'nBTC=', self.nBTC, 'total_wealth=', self.total_wealth(priceData[ts])
                # buy one
                self.nBTC += 1
                self.totalCash -= priceData[ts]
        #sell = (priceData[ts] - pricePrediction)> 0.3 # sell if current price produces profit # TODO account for fees 
        #buy = (pricePrediction - priceData[ts])> 0.3 # buy if the price will go up 
        """
        buy = pricePrediction > priceData[ts]
        sell = pricePrediction < priceData[ts]
        """
        #pdb.set_trace()
        #print 'bought', self.boughtAt, 'cur', priceData[ts], 'pred', pricePrediction, 'sell', sell, 'buy', buy, 'nBTC', self.nBTC, 'maxnBTC', self.maxnBTC
        #if sell and self.nBTC > 0: 
            #self.nBTC -= 1 
            #self.income += priceData[ts]
            #print 'cur', priceData[ts], 'boughtAt', self.boughtAt + 0.5, 'pricePrediction', pricePrediction, 'nBTC', self.nBTC, 'sold 1 bitcoin at {0}, boughAt={1}'.format(priceData[ts],  self.boughtAt)
            #return ('sell',priceData[ts])
        #elif buy and self.nBTC < self.maxnBTC: 
            #self.nBTC += 1 
            #self.invested += priceData[ts]
            #self.boughtAt = ((self.nBTC * self.boughtAt) + priceData[ts]) / float(self.nBTC + 1)
            #print 'cur', priceData[ts], 'boughtAt', self.boughtAt + 0.5, 'pricePrediction', pricePrediction, 'nBTC', self.nBTC,  'bought 1 bitcoin at {0}'.format(priceData[ts])
            ##pdb.set_trace()
            #return ('buy',priceData[ts])
        #print 'cur', priceData[ts], 'boughtAt', self.boughtAt + 0.5, 'pricePrediction', pricePrediction, 'nBTC', self.nBTC, '(neither)'

        return None


    # return percent profit
    def simulate(self, ts_range, priceData): 
        startWealth = self.total_wealth(priceData[ts_range[0]])
        initialBTCWealth = priceData[ts_range[0]] * self.nBTC
        print 'initialBTCWealth = {0} * {1} = {2}'.format(priceData[ts_range[0]] , self.nBTC, initialBTCWealth)

        # get price predcitions
        actions = [] 
        for ts in ts_range: 
            if (ts not in priceData) or (ts + self.timestep not in priceData):
                continue
            if self.predictionMethod == 'perfectRandom': 
                if random.random() > 0.5: 
                    pricePrediction = priceData[ts + self.timestep] + random.random()# 
                else: 
                    pricePrediction = priceData[ts + self.timestep] + -1 * random.random()# 
            elif self.predictionMethod == 'perfect':
                pricePrediction = priceData[ts + self.timestep]
            else: 
                net = NeuralNetwork.NeuralNetwork(ts, priceData, 24, 10, 200)
                pricePredition = net.predictPrice(ts, 1, priceData) # ERROR :(
            action = self.simulate_ts(ts, pricePrediction, priceData)
            if action is not None: 
                actions.append(action)
        currentBtcWealth = self.actualPrices[ts_range[len(ts_range) - 1]] * self.nBTC
        profit = float(self.income + currentBtcWealth) / (self.invested + initialBTCWealth)
        #print 'initialBTC: {0}, nBTC: {1}, invested: {2}, income: {3}, profit%: {4}'.format(self.initialBTC, self.nBTC, self.invested, self.income, profit)
        profit = (self.totalCash + currentBtcWealth) / float(initialBTCWealth)
        profit = self.total_wealth(self.actualPrices[ts_range[len(ts_range) - 1]])/ startWealth 
        return {'profit': profit, 'nBTC': self.nBTC, 'cash': self.totalCash}
        #return {'profit':profit, 'nBTC': self.nBTC, 'invested': self.invested, 'income': self.income}
    # min_ts is the minimum timestamp we will ever examine (e.g. 1387174080 for after the crash)
    def randomSimulate(self,timestep = 60, ntimes=60*24, n=100, min_ts = None):
        profits = {} 
        self.nBTC = 3
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
        return profits, np.mean([i['profit'] for i in profits.values() if (not math.isinf(i['profit']) and (not math.isnan(i['profit'])))])

    #def simulateSinceTs(self, timestep = 3600, start_ts = 1387174080, tsPerRun= 24):
    #for ts_start in xrange(start_ts, max(priceData.keys()), tsPerRun):

priceData = pickle.load(open('../data/bitcoin_prices.pickle'))

test = RuleBasedActionPicker(nBTC = 3, boughtAt = 300, maxnBTC = 10, buySellStep = 1, actualPrices = priceData, timestep = 60, predictionMethod = 'perfect')
profits = test.randomSimulate(timestep=3600,ntimes = 24 * 30 , n=100, min_ts = 1387174080) # 1.0158806245527092
pdb.set_trace()


# second index of profits is the avg. profit over 100 random examples after dec 13 2013.

