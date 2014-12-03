import pickle, dataFetcher, BitcoinCSP, CSP, math, CSPsolver, pdb

class BitcoinCSP: 

	# pricePredictions
	def __init__(self, pricePredictions, nBTC, boughtAt, prevPrice):
		self.pricePredictions = pricePredictions
		self.nBTC = nBTC
		timesteps = sorted(pricePredictions.keys())
		timestep = abs(timesteps[1] - timesteps[0])
		self.csp = CSP.CSP()
		self.prevPrice = prevPrice
		for ts in pricePredictions.keys(): 
			domain = range(0,nBTC+1)
			self.csp.add_variable(ts,range(0,nBTC+1))
			#it's better to assign a purchase to later because we dont' want to assign all
			# purchases prematurely.
			timeElapsed = float((ts - min(pricePredictions.keys())) / timestep)
			def earlierIsBetter(val): 
				return 1.0/math.log(timeElapsed + 2) 
			def higherPriceBetter(val): 
				if pricePredictions[ts] < boughtAt and val > 0: 
					return 0
				if pricePredictions[ts] < boughtAt or val == 0:  
					return 1 
				else:	
					print 'returning {0} for {1}'.format(pricePredictions[ts] - boughtAt , ts)
					if (pricePredictions[ts] - boughtAt) > 1: 
						return 1 + (pricePredictions[ts] - boughtAt)**val
					else:
						return 1 + val * (pricePredictions[ts] - boughtAt)
			def increasingDecreasing(val):
				curPrice = pricePredictions[tss]
				if ts > min(pricePredictions.keys()): 
					prevPrice = pricePredictions[ts - timestep]
				else: 
					prevPrice = self.prevPrice
				if prevPrice >= curPrice and val > 0: # decreasing
					return curPrice/float(prevPrice)
				elif prevPrice <= curPrice and val > 0:  # increasing
					return curPrice/float(prevPrice)
				return 1 
			self.csp.add_unary_potential(ts, earlierIsBetter)
			self.csp.add_unary_potential(ts, increasingDecreasing)
			self.csp.add_unary_potential(ts, higherPriceBetter)
		def sum_is_ok(sum): 
			if sum > self.nBTC: return 0 
			else: return 1 
		sumVar = self.get_sum_variable('sellMax', pricePredictions.keys(), self.nBTC)
		self.csp.add_unary_potential(sumVar, sum_is_ok)

	def solve(self):
		search = CSPsolver.BacktrackingSearch()
		search.solve(self.csp)
		for k in search.optimalAssignment.keys(): 
			if search.optimalAssignment[k] > 0 and str(k.__class__) not in ("<type 'tuple'>","<type 'str'>"): 
				print 'sell {0} bitcoin at {1} for {2}'.format(search.optimalAssignment[k], k, self.pricePredictions[k])
		pdb.set_trace()

	def get_sum_variable(self, name, variables, maxSum):

    # Problem 2b
    # reutrn a variable that's consistent only when it's the second index of a3
    # need to generalize this 

	    def start_with_0(A1_val): 
	        return int(A1_val[0] == 0)

	    def X_A_consistency(X_val, A_val): 
	        return int(A_val[1] == X_val + A_val[0])

	    def Ai_Aj_consistency(Ai_val, Aj_val): 
	        return int(Aj_val[0] == Ai_val[1])

	    def sum_matches_last(A_last_val, sum_val): 
	        return sum_val == A_last_val[1]

	    domain = []
	    for i in range(maxSum + 1):
	        for j in range(i,maxSum + 1): 
	            domain.append([i,j])

	    for var_i in xrange(len(variables)):
	        var = variables[var_i]
	        new_var = (var, 'counter') 
	        self.csp.add_variable(new_var, domain)
	        self.csp.add_binary_potential(var, new_var, X_A_consistency)
	        if var_i != 0: 
	            prev_var = (variables[var_i - 1], 'counter')
	            self.csp.add_binary_potential(prev_var, new_var, Ai_Aj_consistency)
	            #print 'csp.add_binary_potential({0}, {1}, Ai_Aj_consistency)'.format(prev_var, new_var)
	        else: 
	            #print 'csp.add_binary_potential({0}, start_with_0)'.format(new_var)
	            self.csp.add_unary_potential(new_var, start_with_0)
	        if var_i == len(variables) - 1: 
	            self.csp.add_variable(name, range(0, maxSum + 1))
	            #print 'csp.add_binary_potential({0}, {1}, sum_matches_last)'.format(new_var, name)
	            self.csp.add_binary_potential(new_var, name, sum_matches_last)
	        #print 'csp.add_binary_potential({0}, {1}, X_A_consistency)'.format(var, new_var)
	    return name 



"""
prices = {1413237600: 400, 1413252000: 450}
import pickle, dataFetcher, BitcoinCSP
#priceData = pickle.load(open('../data/bitcoin_prices.pickle'))
#prices24 = dataFetcher.aggregated_prices(priceData, 1413590400, 24, 3600,'hash' )
prices24 = {1413504000: 378.54483333333337, 1413532800: 376.85616666666664, 1413568800: 376.90699999999998, 1413597600: 376.178, 1413626400: 379.63516666666663, 1413655200: 380.37383333333332, 1413511200: 378.66600000000005, 1413547200: 380.77499999999998, 1413576000: 376.28866666666664, 1413604800: 375.45466666666675, 1413633600: 380.88550000000004, 1413662400: 380.91816666666665, 1413518400: 379.62283333333329, 1413540000: 377.80383333333333, 1413554400: 380.02983333333327, 1413583200: 373.71899999999994, 1413612000: 375.14399999999995, 1413640800: 379.79616666666664, 1413669600: 381.14983333333328, 1413525600: 377.12666666666667, 1413561600: 378.70796610169492, 1413590400: 374.9516666666666, 1413619200: 375.86149999999992, 1413648000: 379.50849999999997}

bCSP = BitcoinCSP.BitcoinCSP(prices24,2,378, 378) 
bCSP.solve()

"""