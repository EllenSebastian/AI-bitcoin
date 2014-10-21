import time

time_format = '%d/%m/%Y %H:%M:%S'

# return 2d array of data in file 
def read_data(file): 
	return [[time.strptime(line.strip().split(',')[0], time_format), float(line.strip().split(',')[1])]
		 for line in open(file)]

