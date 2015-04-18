import RuleBasedActions, pickle, pdb, dataFetcher
priceData = pickle.load(open('../../cs221-stuff/data/bitcoin_prices.pickle'))



maxnbtc = 10
cash = 10000
nbtc = 0
timestep = 3600
predictionMethod = 'neuralnet'
pdb.set_trace()
test = RuleBasedActions.RuleBasedActionPicker(priceData=priceData, nBTC = nbtc, cash=cash, boughtAt = 0, maxnBTC = maxnbtc, buySellStep = 1, actualPrices = priceData, timestep = timestep, predictionMethod = 'neuralnet')
pdb.set_trace()
#profits = test.randomSimulate(timestep=3600,ntimes = 24 * 12   * 30, n=50, min_ts = 1387174080) # 1.0158806245527092
#print 'maxnbtc', maxnbtc, 'cash', cash, 'nbtc', nbtc, 'timestep', 3600, 'predictionMethod', predictionMethod, 'profit', profits[1]

#pdb.set_trace()


## test from dec 16 until present
ts_range = range(1387174080, max(priceData.keys()) - 7200 - 169740 - 540, timestep)
price_subset = dataFetcher.aggregated_prices(priceData, max(ts_range), len(ts_range) + 3 ,timestep, 'hash')
all_profit = test.simulate(ts_range, price_subset)
print 'maxnbtc', maxnbtc, 'cash', cash, 'nbtc', nbtc, 'timestep', 3600, 'predictionMethod', predictionMethod, 'profit', all_profit
pdb.set_trace()
#print all_profit[1]

maxnbtc = 100
cash = 1000
nbtc = 0
timestep = 3600
predictionMethod = 'neuralnet'
test = RuleBasedActionPicker(nBTC = nbtc, cash=cash, boughtAt = 0, maxnBTC = maxnbtc, buySellStep = 1, actualPrices = priceData, timestep = timestep, predictionMethod = 'neuralnet')
ts_range = range(1387174080, max(priceData.keys()) - 7200 - 169740 - 540, timestep)
price_subset = dataFetcher.aggregated_prices(priceData, max(ts_range), len(ts_range) + 3 ,timestep, 'hash')
all_profit = test.simulate(ts_range, price_subset)
print 'maxnbtc', maxnbtc, 'cash', cash, 'nbtc', nbtc, 'timestep', timestep, 'predictionMethod', predictionMethod, 'profit', all_profit
pdb.set_trace()
"""
# second index of profits is the avg. profit over 100 random examples after dec 13 2013.

with perfect predictions: 
monthly with timestep = 1 hour: 
profits starting out with 3 btc and 0 cash, limit 10 btc: 1.78
profits starting out with 0 btc an 1000 cash, limit 10 btc: 2.23, 2.18
profits starting wiht 3 btc and 10000 cash, limit 10 btc: 1.15 (due to too much cash to start with) 
from dec16, 0btc $500 limit 100,40.303027323959078, 20149.926162
from dec16, 5btc $500 limit 100,5.7676382906534673, 27698.115412


with neuralnet predictions; 
monthly profit $1000 limit 100,
monthly profit $10000 limit 100, 

from dec16, 0btc $500 limit 100, 5645.47106417, 11.301102128346882
from dec16, 5btc $500 limit 100, 23838.065193, 4.4996678884859431

from dec16, 0btc $1000 limit 100 timestep 60, 14352.2598852, 14.365277385188294
from dec16, 0btc $1000 limit 100 timestep 60
from dec16, 0btc $1000 limit 100 timestep 600,23454.426365833409, 23.798569365833409



maxnbtc 1 cash 1000 nbtc 0 timestep 600 predictionMethod neuralnet profit {'profit': 23.798569365833409, 'nBTC': 1, 'cash': 23454.426365833409, 'predAccuracy': 0.0}
> /Users/ES/Desktop/cs221/project/AI-bitcoin/p-progress/RuleBasedActions.py(221)<module>()
-> from dec16, 0btc $1000 limit 5, 8915.9727038, 8.9147042037981379

"""


#ts = 1403610960
#net = NeuralNetwork.NeuralNetwork(endTimeStamp=ts + 50*3600,windowSize=24, numFeatures=10, numDataPoints=500, frequency=3600)
#pricePredition = net.predictPrice(ts, 1) # ERROR :(
