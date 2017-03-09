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
    # clf = tree.DecisionTreeClassifier()
    # min_impurity_split=50 adress overfitting
    
    # clf = tree.DecisionTreeClassifier()
    # accuracy - 0.908
    
    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    # accuracy - 0.912 - my data from local Eclipse IDE
    # accuracy - 0.908 - online udacity.com decision Tree Accuracy Quiz - probably different data 
    
    #clf = tree.DecisionTreeClassifier(min_samples_split=50)
    # accuracy - 0.912
    # summary as min_samples_split increases so does accuracy

    
    # create, instantiate the sklearn.tree.tree.DecisionTreeClassifier'
    clf = clf.fit(features_train, labels_train)
    
    return clf
