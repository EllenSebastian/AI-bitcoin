import requests
import json
import datetime
import collections
import csv

'''
Crawl transaction details based on block information and group them on hourly-basis
'''

block_hash = '00000000000000000ea75faa591d2fad17f12bd68403e2546c0394217ad617e2'
data = collections.Counter()
for i in xrange(1000):
    res = requests.get('http://blockexplorer.com/rawblock/' + block_hash)
    res_json = json.loads(res.content)
    timestamp = int(res_json['time'])
    data[timestamp - timestamp % 3600] += len(res_json['tx'])
    block_hash = res_json['prev_block']
data = [item for item in data.items()]
data = sorted(data, key=lambda ele: ele[0])
print data
with open('transactionCount', 'w') as output:
    for timestamp, value in data:
        output.write('%d %d\n' % (timestamp, value))
