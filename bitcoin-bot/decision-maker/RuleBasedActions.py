# sell at an observed trough (some decreases in the past and a predicted increase)
# sell whenever you see a price higher than what you bought at 
# here, btc refers to the "other" currency (not the one you started with).  and dollars refers to the currency you started with. 
# actualPrices is a hash between timestamps and the price of bitcoins in terms of dollars.
import NeuralNetwork, random, pickle, pdb, dataFetcher, linecache, math
import numpy as np
class RuleBasedActionPicker: 
    def __init__(self, priceData, nBTC, cash, boughtAt, maxnBTC, buySellStep, actualPrices, timestep = 60, predictionMethod='NeuralNet', minProfit = 0.5):
        self.priceData = priceData
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
        self.totalCash = cash

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
            if self.totalCash >= priceData[ts] and self.nBTC < self.maxnBTC:
                print 'buy 1 bitcoin at p=', priceData[ts], 'nBTC=', self.nBTC, 'total_wealth=', self.total_wealth(priceData[ts]), 'ts',ts
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
    def simulate(self, ts_range, priceData=None):
        if priceData is not None: 
            self.priceData = priceData 
        pred = [] 
        actual = [] 
        err = [] 
        actualP = [] 
        predP = [] 
        errP = [] 
        initialWealth = self.priceData[ts_range[0]] * self.nBTC + self.totalCash
        #print 'initialBTCWealth = {0} * {1} = {2}'.format(priceData[ts_range[0]] , self.nBTC, initialBTCWealth)
        fn, fp, tn, tp = 0,0,0,0
        # get price predcitions
        actions = [] 
        net = NeuralNetwork.NeuralNetwork(priceData=self.priceData, endTimeStamp=max(ts_range),windowSize=24,numFeatures=10,numDataPoints=len(ts_range) + 500,frequency=self.timestep)
        for ts in ts_range: 
            if (ts not in self.priceData) or (ts + self.timestep not in self.priceData):
                print 'ts not in priceData'
                continue
            if self.predictionMethod == 'perfectRandom': 
                if random.random() > 0.5: 
                    pricePrediction = self.priceData[ts + self.timestep] + random.random()# 
                else: 
                    pricePrediction = self.priceData[ts + self.timestep] + -1 * random.random()# 
            elif self.predictionMethod == 'perfect':
                pricePrediction = self.priceData[ts + self.timestep]
            else: 
                #net = NeuralNetwork.NeuralNetwork(endTimeStamp=ts+50*self.timestep,windowSize=24,numFeatures=10,numDataPoints=500,frequency=self.timestep)
                res = net.predictPrice2(ts, 1) # ERROR :(
                pricePrediction = res[0].values()[0]
                predictedPriceDiff = (pricePrediction - self.priceData[ts]) / self.priceData[ts]
                actualPriceDiff =( self.priceData[ts + self.timestep] - self.priceData[ts]) / self.priceData[ts]
                pred.append(predictedPriceDiff)
                actual.append(actualPriceDiff)
                predP.append(pricePrediction)
                actualP.append(self.priceData[ts + self.timestep])
                err.append(predictedPriceDiff - actualPriceDiff)
                errP.append(pricePrediction - self.priceData[ts + self.timestep])
                if actualPriceDiff < 0:
                    if predictedPriceDiff < 0: 
                        tn += 1
                    else:
                        fp += 1
                elif actualPriceDiff > 0: 
                    if predictedPriceDiff < 0: 
                        fn += 1
                    else:
                        tp += 1    
            action = self.simulate_ts(ts, pricePrediction, self.priceData)
            if action is not None: 
                actions.append(action)
        pdb.set_trace()
        try: 
            currentBtcWealth = self.priceData[ts_range[len(ts_range) - 2]] * self.nBTC
        except Exception, e:
            pdb.set_trace()
        #print 'initialBTC: {0}, nBTC: {1}, invested: {2}, income: {3}, profit%: {4}'.format(self.initialBTC, self.nBTC, self.invested, self.income, profit)
        profit = (self.totalCash + currentBtcWealth) / float(initialWealth)
        print 'tp',tp,'tn',tn,'fp',fp,'fn',fn, profit,'profit'
        return {'profit': profit, 'nBTC': self.nBTC, 'cash': self.totalCash, 'predAccuracy': float(tp + tn) / float(tp+tn+fp+fn + 0.01)} 
        #return {'profit':profit, 'nBTC': self.nBTC, 'invested': self.invested, 'income': self.income}
        # min_ts is the minimum timestamp we will ever examine (e.g. 1387174080 for after the crash)
    """
plt.plot(ts_range[0:7863], pred,'b')
plt.plot(ts_range[0:7863], actual,'r')
plt.title('Actual vs. predicted price changes over time')
plt.ylim(-0.2,.2)
plt.show()

plot.plt(ts_range[0:7863], err)
plt.title('Predicted price change error over time')

plt.plot(ts_range[0:7863], predP,'b')
plt.plot(ts_range[0:7863], actualP,'r')
plt.title('Actual vs. predicted prices over time')
plt.show()

plt.plot(ts_range[0:7863], errP)
plt.title('Predicted price error over time')
plt.show()

    """ 
    def randomSimulate(self,timestep = 60, ntimes=60*24, n=100, min_ts = None):
    	cashReset = self.totalCash
    	btcReset = self.nBTC
        profits = {} 
        self.timestep = timestep
        if min_ts is None: 
            min_ts = min(self.actualPrices.keys())
        allowedTs = [i for i in self.actualPrices.keys() if i > (min_ts + (timestep + 1) * ntimes)]
        for j in xrange(n):
            self.totalCash = cashReset
            self.nBTC = btcReset
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
