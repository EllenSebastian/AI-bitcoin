# Baseline algorithm. Take past five days average price to predict.
testing_size = 100
with open('./data/market-price.txt', 'r') as pricefile:
    data = []
    for line in pricefile:
        date, price = line.strip().split(',')
        price = float(price)
        data.append((date, price))
    squared_error = 0
    for i in xrange(len(data) - testing_size, len(data)):
        predict_price = sum([price for date, price in data[i-5:i]]) / 5.0
        squared_error += (predict_price - data[i][1])** 2
    mse = squared_error / testing_size

print mse
