import csv
import datetime
from pylab import *

'''
compare the correlation between #transaction and $price on hourly-basis
'''

time1 = []
price = []
with open('per_hour_monthly_sliding_window.csv', 'rbU') as csvfile:
    reader = csv.reader(csvfile)
    init = True
    for time, _, _, avg in reader:
        if init:
            init = False
            continue
        timestamp = int(datetime.datetime.strptime(time, '%m/%d/%y %H:%M').strftime("%s"))
        time1.append(timestamp - timestamp % 3600)
        price.append(avg)

time2 = []
counts = []
with open('transactionCount', 'r') as countfile:
    for line in countfile:
        time, count = line.split()
        time2.append(time)
        counts.append(count)

plot(time1, price, 'g', time2, counts, 'r')
xlabel('timestamp')
ylabel('price')
grid(True)
show()
