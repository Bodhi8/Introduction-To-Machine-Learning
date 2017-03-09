'''
Created on Mar 8, 2017

@author: Menfi
git add classiftDT.py
git commit classiftDT.py -m " DT - Decision Tree - Supervised Classification Algorithm"
'''

from sklearn import tree

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifier
    
    # create, instantiate the sklearn.tree.tree.DecisionTreeClassifier'
    clf = tree.DecisionTreeClassifier()
    
    # create, instantiate the sklearn.tree.tree.DecisionTreeClassifier'
    clf = clf.fit(features_train, labels_train)
    
    return clf
