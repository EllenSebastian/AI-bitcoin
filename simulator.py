from sklearn.linear_model import SGDClassifier

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 2])
clf = linear_model.SGDClassifier()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))

