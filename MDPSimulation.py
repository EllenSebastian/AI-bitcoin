import pickle
from sklearn.gaussian_process import GaussianProcess
import BitcoinMDP

# per minute price change
prices_change = pickle.load(open('./data/small_bitcoin_price_change.pickle'))
prices = pickle.load(open('./data/small_bitcoin_prices.pickle'))

shrinked_prices = prices_change[-9000:]
shrinked_len = len(shrinked_prices)
train_prices = [p for t, p in shrinked_prices[: 2 * shrinked_len / 3]]
test_prices = [p for t, p in shrinked_prices[-shrinked_len / 3:]]
test_timestamp = [t for t, p in shrinked_prices[-shrinked_len / 3:]]

a = 0
max_p = 0
min_p = 0
total_time = 120
start_time = 330
test_range = range(start_time, start_time + total_time)
for idx in test_range:
    a = a + test_prices[idx]
    if a > max_p:
        max_p = a
    if a < min_p:
        min_p = a
print max_p, min_p

span = 30
train_X = []
train_Y = []
for idx in range(len(train_prices)):
    if idx < span:
        continue
    train_X.append(train_prices[(idx-span):idx])
    train_Y.append(train_prices[idx])

gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(train_X, train_Y)

print 'finished training GP'

min_p = int(round(min_p*10))
max_p = int(round(max_p*10))
print [min_p, max_p]
mdp = BitcoinMDP.BitcoinMDP(total_time, 10, 0, 1, [min_p, max_p], 5)
vio = BitcoinMDP.ValueIteration()
vio.solve(mdp)
#print vio.pi
print 'finished solving MDP'
test_X = []
test_Y = []
for idx in test_range:
    test_X.append(test_prices[(idx-span):idx])
    test_Y.append(test_prices[idx])
Y, Sigma2 = gp.predict(test_X, eval_MSE=True)
print Y, Sigma2
current_time = total_time
btc = 10
print 'start timestamp:', test_timestamp[start_time]
initial_wealth = prices[test_timestamp[start_time]] * btc
p_change = 0
income = 0
for idx in range(len(Y)):
    #print p_change, round((Y[idx] + p_change)*10)
    round_y = max(min(int(round((Y[idx] + p_change) * 10)), max_p), min_p)
    round_sigma = min(int(round(Sigma2[idx]**0.5 * 10)), 5)
    #round_y = max(min(int(round((test_Y[idx] + p_change) * 10)), max_p), min_p)
    #round_sigma = 0
    current_state = (current_time, btc, round_y, round_sigma)
    action = vio.pi[current_state]
    print 'timestamp:', test_timestamp[test_range[idx]]
    print 'price:', prices[test_timestamp[test_range[idx]]]
    print 'state:', current_state, 'action:', action
    action = min(action, btc)
    income += prices[test_timestamp[test_range[idx]]] * action
    print 'income:', income
    current_time -= 1
    btc = max(btc - action, 0)
    p_change += test_Y[idx]
    if btc == 0:
        break

print initial_wealth, income, (income - initial_wealth) / initial_wealth
