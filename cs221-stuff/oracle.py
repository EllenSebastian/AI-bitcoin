
# oracle algorithm: SGD on past data including price, future data 
# for all except price.
import numpy as np
import datetime, copy, util, time
from datetime import date
from datetime import timedelta
from datetime import datetime
from sklearn import linear_model
from datetime import timedelta
from sklearn.cross_validation import KFold

non_price_inputs =  ['avg-confirmation-time.txt', 'estimated-transaction-volume.txt', 'my-wallet-transaction-volume.txt', 'total-bitcoins.txt', 
             'bitcoin-days-destroyed-cumulative.txt','hash-rate.txt', 'n-orphaned-blocks.txt','trade-volume.txt', 'bitcoin-days-destroyed.txt','market-cap.txt', 
             'n-transactions-excluding-popular.txt','transaction-fees.txt', 'blocks-size.txt','n-transactions-per-block.txt', 'tx-trade-ratio.txt', 
             'cost-per-transaction.txt','miners-revenue.txt', 'n-transactions.txt', 'difficulty.txt','my-wallet-n-tx.txt', 'n-unique-addresses.txt', 
             'estimated-transaction-volume-usd.txt', 'my-wallet-n-users.txt', 'output-volume.txt']

data = {}
for f in non_price_inputs: 
    data[f] = util.read_data('data/' + f)

data['market-price.txt'] = util.read_data('data/market-price.txt')

day_price = util.day_to_price()


# find all the features and prices for 100-600 days before today
all_features = []
Y = []

for test_day in range(100,600): # the last 600-100 days from today
    end_day = date.today() - timedelta(days = test_day + 1)
    start_day = date.today() - timedelta(days = test_day + 101) # look at the last 100 days
    this_day = date.today() - timedelta(days = test_day)
    future_day = this_day + timedelta(days = 200)
    features = []
    for file in non_price_inputs:
        cur_features = util.make_feature_vector_from_file(data[file], start_day, future_day) 
        features += cur_features 
    features += util.make_feature_vector_from_file(data['market-price.txt'], start_day, end_day)
    if this_day in day_price: 
        all_features.append(features)
        Y.append(day_price[this_day])


# limit the output to examples where there are no Nones (no missing days)
all_features_clean, Y_clean = util.remove_nones(all_features, Y)


### LOOCV has mse=34, so use training error
kf = KFold(len(all_features_clean), n_folds = len(all_features_clean))
total_error = 0
for train,test in kf: 
    print test
    this_x = []
    this_y = []
    for i in train: 
        this_x.append(all_features_clean[i])
        this_y.append(Y_clean[i])
    reg = linear_model.LinearRegression()
    reg.fit(this_x, this_y)
    predicted = reg.predict(all_features_clean[test[0]])
    squared_error = (predicted - Y_clean[test[0]])**2
    total_error += squared_error


print total_error
print total_error / len(all_features_clean)



# training error is 0 

reg = linear_model.LinearRegression()
reg.fit(all_features_clean, Y_clean)

total_error = 0
for i in xrange(len(all_features_clean)):
    predicted = reg.predict(all_features_clean[i])
    total_error += (predicted - Y_clean[i])**2

mse = total_error / len(all_features_clean)

print mse

