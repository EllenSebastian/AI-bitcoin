import pickle, dataFetcher, BitcoinCSP, CSP, math, CSPsolver, pdb

class BitcoinCSP: 

	# pricePredictions
	def __init__(self, pricePredictions, nBTC, boughtAt):
		self.pricePredictions = pricePredictions
		self.nBTC = nBTC
		timesteps = sorted(pricePredictions.keys())
		timestep = abs(timesteps[1] - timesteps[0])
		self.csp = CSP.CSP()
		for ts in pricePredictions.keys(): 
			domain = range(0,nBTC+1)
			self.csp.add_variable(ts,range(0,nBTC+1))
			#it's better to assign a purchase to later because we dont' want to assign all
			# purchases prematurely.
			timeElapsed = float((ts - min(pricePredictions.keys())) / timestep)
			#def earlierIsBetter(val): return 1.0/math.log(timeElapsed + 2) 
			#self.csp.add_unary_potential(ts, earlierIsBetter)
			def higherPriceBetter(val): 
				if pricePredictions[ts] < boughtAt: 
					return 0
				elif val == 0:  
					return 1 
				else:	
					print 'returning {0} for {1}'.format(pricePredictions[ts] - boughtAt , ts)
					if (pricePredictions[ts] - boughtAt) > 1: 
						return 1 + (pricePredictions[ts] - boughtAt)**val
					else:
						return 1 + val * (pricePredictions[ts] - boughtAt)

			self.csp.add_unary_potential(ts, higherPriceBetter)
		def sum_is_ok(sum): 
			if sum > self.nBTC: return 0 
			else: return 1 
		sumVar = self.get_sum_variable('sellMax', pricePredictions.keys(), self.nBTC)
		self.csp.add_unary_potential(sumVar, sum_is_ok)

	def solve(self):
		search = CSPsolver.BacktrackingSearch()
		search.solve(self.csp)
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
#prices = dataFetcher.aggregated_prices(priceData, 1413590400, 1413590400 + 24 * 60 * 60, 3600,'hash' )
bCSP = BitcoinCSP.BitcoinCSP(prices,2,390) 
bCSP.solve()

"""