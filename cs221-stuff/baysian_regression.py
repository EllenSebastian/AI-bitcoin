import numpy as np
import pickle, math, time
from sklearn import linear_model
from sklearn import cluster

prices = pickle.load(open('./data/small_bitcoin_price_change.pickle'))

shrinked_prices = prices[-30000:]
shrinked_len = len(shrinked_prices)
ts_prices = [p for t, p in shrinked_prices[:shrinked_len / 3]]
train_prices = [p for t, p in shrinked_prices[shrinked_len / 3: 2 * shrinked_len / 3]]
test_prices = [p for t, p in shrinked_prices[-shrinked_len / 3:]]

#print ts_prices
#print train_prices
#print test_prices

# generate time series s1, s2, s3
basic_span = 30
s1 = []
s2 = []
s3 = []
for idx in range(len(ts_prices)):
    if idx >= basic_span:
        s1.append(((np.array(ts_prices[(idx - basic_span):idx])), ts_prices[idx]))
    if idx >= 2 * basic_span:
        s2.append(((np.array(ts_prices[(idx - 2 * basic_span):idx])), ts_prices[idx]))
    if idx >= 4 * basic_span:
        s3.append(((np.array(ts_prices[(idx - 4 * basic_span):idx])), ts_prices[idx]))

# run k-means to pick top patterns
def get_effective_patterns(s, assignment, num=20):
    clusters = {}
    for idx in range(len(s)):
        if assignment[idx] not in clusters:
            clusters[assignment[idx]] = []
        clusters[assignment[idx]].append(s[idx])
    stds = []
    for k, l in clusters.items():
        #if k == 2 or k == 3 or k == 1:
            #print l
        if len(l) == 1:
            continue
        std = np.std([y for x, y in l])
        stds.append((k, std))
    stds = sorted(stds, key=lambda x: x[1])
    stds = stds[:num]
    #print stds
    res = []
    for k, _ in stds:
        c_mean = np.mean(clusters[k], axis=0)[0]
        min_dist = 100000
        for x, y in clusters[k]:
            dist = np.linalg.norm(x - c_mean)
            if dist < min_dist:
                min_dist = dist
                best_x = x
                best_y = y
        res.append((best_x, best_y))
    return res

km = cluster.KMeans(n_clusters=200)
assignment = km.fit_predict([x for x, y in s1])
s1 = get_effective_patterns(s1, assignment)

km = cluster.KMeans(n_clusters=200)
assignment = km.fit_predict([x for x, y in s2])
s2 = get_effective_patterns(s2, assignment)

km = cluster.KMeans(n_clusters=200)
assignment = km.fit_predict([x for x, y in s3])
s3 = get_effective_patterns(s3, assignment)

#print len(s1), len(s2), len(s3)

# train
def compute_delta_p(s, x):
    numerator = 0
    denominator = 0
    x_np = np.array(x)
    for xi, yi in s:
        diff = x_np - xi
        tmp = math.exp(-np.inner(diff, diff) / 4)
        numerator += tmp * yi
        denominator += tmp
    return numerator / denominator

x_train = []
y_train = []
print time.time()
for idx in range(len(train_prices)):
    if idx < 4 * basic_span:
        continue
    delta_p1 = compute_delta_p(s1, train_prices[(idx - basic_span):idx])
    delta_p2 = compute_delta_p(s2, train_prices[(idx - 2 * basic_span):idx])
    delta_p3 = compute_delta_p(s3, train_prices[(idx - 4 * basic_span):idx])
    x_train.append([delta_p1, delta_p2, delta_p3])
    y_train.append(train_prices[idx])
print time.time()

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

# test
x_test = []
y_test = []
for idx in range(len(test_prices)):
    if idx < 4 * basic_span:
        continue
    delta_p1 = compute_delta_p(s1, test_prices[(idx - basic_span):idx])
    delta_p2 = compute_delta_p(s2, test_prices[(idx - 2 * basic_span):idx])
    delta_p3 = compute_delta_p(s3, test_prices[(idx - 4 * basic_span):idx])
    x_test.append([delta_p1, delta_p2, delta_p3])
    y_test.append(test_prices[idx])
res = lr.predict(x_test)
tp = tn = fp = fn = ignore = 0
threshold = 0
for idx in range(len(res)):
    #print res[idx], y_test[idx]
    if res[idx] > threshold and y_test[idx] > 0:
        tp += 1
    elif res[idx] < -threshold and y_test[idx] < 0:
        tn += 1
    elif res[idx] > threshold and y_test[idx] < 0:
        fp += 1
    elif res[idx] < -threshold and y_test[idx] > 0:
        fn += 1
    else:
        ignore += 1
print 'true pos, true neg, false pos, false neg', tp, tn, fp, fn
print ignore
print (tp + tn) / float(len(res) - ignore)
