import numpy as np   
from sklearn.naive_bayes import GaussianNB
# import sklearn works also
# from sklearn import naive_bayes works too

print('Begin gaussianNaiveBayes Python Module\n')

# Create training points - the Features
X = np.array([ [-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2] ])
print("X is -")
print(X)
# print("\ntype(X) is {}\n".format(type(X))) # numpy.ndarray

# Create training points - the Labels
# These are the knowns
Y = np.array([1, 1, 1, 2, 2, 2])
print("\nY is -")
print(Y)
# print("\ntype(Y) is {}\n".format(type(Y))) # numpy.ndarray

# Create classifier - clf - classifier 
# clf = naive_bayes.GaussianNB() works also
clf = GaussianNB()

# call the fit function of the newly created classifier clf 
# fit or train the clf classifier, clf, classifier learns patterns 
# fit or train, instructor - "give it the training data", instructor - "it learns the pattern"
# X - Features 
# Y - Labels 
# clf.fit(features, labels    )
clf.fit(X, Y)

# clf, classifier uses the patterns learned - fit function, to predict the label or class associated with the new point
# ask the trained classifier, clf for predictions
# input new point(s), point(s) not previously defined in X Features 
# output the label or the class the unknown point(s) belong to
# pred = clf.predict(features_test) *** instructor notes ***
print("\nclf.predict([[-0.8, -1]]) is ")
print(clf.predict([[-0.8, -1]]))
print(clf.predict([[-0.8, -1],[2, 3]]))
print(clf.predict([[2, 3]]))

print('\nEnd gaussianNaiveBayes Python Module')
