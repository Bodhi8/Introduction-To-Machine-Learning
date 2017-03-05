'''
Created on Mar 3, 2017

@author: Menfi
'''

import sklearn.svm

print('\nBegin supportVectorMachines.py Module\n')

# classifier training features
X = [[0, 0], [1, 1]]
print("type(X - training features) - {}".format(type(X)))
#      type(X - training features) - <class 'list'>

# classifier training labels
y = [0, 1]
print("type(y - training labels) - {}\n".format(type(y)))
#      type(y - training labels) - <class 'list'>

# Create, instantiate Support Vector Machines, SVM, Classifier 
clf = sklearn.svm.SVC()
print("type(clf) - {}\n".format(type(clf)))
#      type(clf) - <class 'sklearn.svm.classes.SVC'>

# fit or train the classifier
clf.fit(X,y)

myPrediction = clf.predict([[2., 2.]])
print("myPrediction - {}".format(myPrediction))
#      myPrediction - [1]
print("type(myPrediction) - {}".format(type(myPrediction)))
#      type(myPrediction) - <class 'numpy.ndarray'>


print('\nEnd supportVectorMachines.py Module')
