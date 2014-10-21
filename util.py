import time

time_format = '%d/%m/%Y %H:%M:%S'

# return 2d array of data in file 
# first index of each row is a Time, second index is a Float
def read_data(file): 
	return [[time.strptime(line.strip().split(',')[0], time_format), float(line.strip().split(',')[1])]
		 for line in open(file)]

# aggregation can be 'average' or 'sum'
# agg_length is an integer of days
# return an array of values from start_ts

def make_feature_vector_from_file(file, start_ts, end_ts, aggregation = None, agg_length = None):
	print 'reading from {0}'.format(file)
	data = read_data(file)
	print 'read from file'
	i = len(data) - 1 
	out = []
	while(True): 
		print i 
		if i < 0: 
			break
		if data[i][0] > end_ts: 
			i -= 1
			continue
		if data[i][0] < start_ts: 
			break
		if aggregation == None: 
			out.append(data[i][1])
			i -= 1 
		elif aggregation == 'average' and i > agg_length: 
			sum = 0 
			for j in range(0, agg_length): 
				sum += data[i][1]
				i -= 1
			out.append(sum / float(agg_length))
		elif aggregation == 'sum' and i > agg_length: 
			sum = 0 
			for j in range(0, agg_length): 
				sum += data[i][1]
				i -= 1
			out.append(sum)
		else: 
			i -= 1 
	return out 
