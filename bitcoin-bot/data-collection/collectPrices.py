"""
usage: 
sh collect-prices.sh frequency outfile outfileLength ntimes
e.g. sh collect-prices.sh 60 btcPrices 10000 20000
"""

import CoinbasePriceCollector, sys, pdb
freq = int(sys.argv[1])
outfile = sys.argv[2]
outfileLength = int(sys.argv[3])
ntimes = int(sys.argv[4])
pc = CoinbasePriceCollector.priceCollector(freq,outfile,outfileLength)
pc.run(ntimes)
