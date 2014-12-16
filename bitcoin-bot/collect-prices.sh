#usage: 
#sh collect-prices.sh frequency outfile outfileLength ntimes
#e.g. sh collect-prices.sh 60 btcPrices 10000 20000

#while true; do 
python collect-prices.py $1 $2 $3 $4
#; done