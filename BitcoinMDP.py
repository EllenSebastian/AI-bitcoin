import math
import MDPutil as util
import numpy as np

def pdf(mean, std, value):
    if std == 0:
        return 0 if mean != value else 1
    u = float(value - mean) / abs(std)
    y = (1.0 / (math.sqrt(2 * math.pi) * abs(std))) * math.exp(-u * u / 2.0)
    return y

class ValueIteration(util.MDPAlgorithm):

    # Implement value iteration.  First, compute V_opt using the methods 
    # discussed in class.  Once you have computed V_opt, compute the optimal 
    # policy pi.  Note that ValueIteration is an instance of util.MDPAlgrotithm, 
    # which means you will need to set pi and V (see util.py).
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        prevV = {}
        for state in mdp.states:
            prevV[state] = 0
        self.V = {}
        self.pi = {}
        error = 10
        while error > epsilon:
            error = 0
            for state in mdp.states:
                self.V[state] = -10000000
                for action in mdp.actions(state):
                    value = 0
                    for newState, prob, reward in mdp.succAndProbReward(state, action):
                        value += prob * (reward + mdp.discount() * prevV[newState])
                    #print state, action, value
                    if value >= self.V[state]:
                        self.V[state] = value
                        self.pi[state] = action
                error = max(error, abs(self.V[state] - prevV[state]))
            print error, epsilon
            prevV = self.V.copy()
        # END_YOUR_CODE

class BitcoinMDP(util.MDP):
    def __init__(self, total_time, initial_bitcoins, start_deviation, price_resolution, price_range, max_deviation):
        self.total_time = total_time
        self.initial_bitcoins = initial_bitcoins
        self.start_deviation = start_deviation
        self.price_resolution = price_resolution
        self.price_range = price_range
        self.max_deviation = max_deviation

    def _round_(self, value):
        t = int(value / self.price_resolution) * self.price_resolution
        return t if value - t < t + self.price_resolution - value else t + self.price_resolution

    def computeStates(self):
        self.states = []
        for time in range(self.total_time + 1):
            for bitcoin in range(self.initial_bitcoins + 1):
                for price in range(self.price_range[0], self.price_range[1] + self.price_resolution, self.price_resolution):
                    for std in range(0, self.max_deviation + 1, self.price_resolution):
                        self.states.append((time, bitcoin, price, std))
        print len(self.states)

    def startState(self):
        return (self.total_time, self.initial_bitcoins, 0, start_deviation)

    def actions(self, state):
        return range(self.initial_bitcoins + 1)

    def succAndProbReward(self, state, action):
        if state[0] == 0:
            return []
        action = min(action, state[1])
        high = self._round_(min(state[2] + 1.96 * state[3], self.price_range[1])) # percentile for CI 
        low = self._round_(max(state[2] - 1.96 * state[3], self.price_range[0]))
        p_range = range(low, high + self.price_resolution, self.price_resolution)
        t_prop = {}
        #print p_range
        for price in p_range:
            t_prop[price] = pdf(state[2], state[3], price)
        # normalize
        sum_prop = sum(t_prop.values())
        for price in p_range:
            t_prop[price] = t_prop[price] / sum_prop
        res = []
        for price in p_range:
            reward = -price * action
            res.append(((state[0] - 1, state[1] - action, price, state[3]), reward, t_prop[price]))
        return res

    def discount(self):
        return 1

#mdp = BitcoinMDP(2, 2, 0, 1, [-2, 2], 1)
#vio = ValueIteration()
#vio.solve(mdp)
#print vio.pi
"""
mdp = BitcoinMDP(3, 3, 0, 1, [-4, 4], 2)
>>>>>>> first pass at CSP
vio = ValueIteration()
vio.solve(mdp)
print vio.pi
"""
