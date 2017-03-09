'''
Created on Mar 8, 2017

@author: Menfi

git add decisionTreeExample.py
git commit decisionTreeExample.py -m "udacity.com - Introduction To Machine Learning Examples - ** Decision Tree ** -  Supervised Classification Algorithm"
'''

# from sklearn import tree
import sklearn.tree

print('\nBegin decisionTreeExample.py Python module\n')

X = [ [0, 0], [1, 1] ]
Y = [0, 1]

# clf = sklearn.tree.DecisionTreeClassifier()
myDecisionTreeClassifierInstance = sklearn.tree.DecisionTreeClassifier()
print("myDecisionTreeClassifierInstance - {}\n".format(myDecisionTreeClassifierInstance))
# print("type(myDecisionTreeClassifierInstance) - {}\n".format(type(myDecisionTreeClassifierInstance)))
#      type(myDecisionTreeClassifierInstance) - <class 'sklearn.tree.tree.DecisionTreeClassifier'>

# train or fit the sklearn.tree.tree.DecisionTreeClassifier using training data
# fit - training features, training labels used
myDecisionTreeClassifierInstance = myDecisionTreeClassifierInstance.fit(X, Y)
print("myDecisionTreeClassifierInstance - {}\n".format(myDecisionTreeClassifierInstance))
# print("type(myDecisionTreeClassifierInstance) - {}\n".format(type(myDecisionTreeClassifierInstance)))
#      type(myDecisionTreeClassifierInstance) - <class 'sklearn.tree.tree.DecisionTreeClassifier'>

# use fitted classifier to make a prediction from test data
# pred = myDecisionTreeClassifierInstance.predict(features_test)  ([[2., 2.]])
pred = myDecisionTreeClassifierInstance.predict([[2., 2.]])
print("pred - {}\n".format(pred))
#      pred - [1]
# print("type(pred) - {}\n".format(type(pred)))
#      type(pred) - <class 'numpy.ndarray'>




 
print('End   decisionTreeExample.py Python module')
