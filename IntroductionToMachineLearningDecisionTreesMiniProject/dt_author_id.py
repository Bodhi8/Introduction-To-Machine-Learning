'''
Created on Mar 11, 2017

@author: Menfi
'''

#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
# from email_preprocess import preprocess
import email_preprocess

import sklearn.tree


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = email_preprocess.preprocess()

print("type(features_train) - {}".format(type(features_train)))
print("features_test[0] - {}".format(features_test[0]))
print("features_test[1] - {}".format(features_test[1]))
print("len(features_test[0]) - {}".format(len(features_test[0])))
print("len(features_test[1]) - {}".format(len(features_test[1])))




#########################################################
### your code goes here ###

t0 = time()

myDecisionTreeClassifier = sklearn.tree.DecisionTreeClassifier(min_samples_split = 40)
myDecisionTreeClassifier.fit(features_train,labels_train)
print("myDecisionTreeClassifier.fit(features_train,labels_train) - fit - training time - {}".format(round(time() - t0, 3)))

myDecisionTreeClassifierAccuracy = myDecisionTreeClassifier.score(features_test,labels_test)
print("myDecisionTreeClassifierAccuracy - {}".format(myDecisionTreeClassifierAccuracy))



#########################################################


print('\nBegin dt_author_id.py Python module\n')
x = 5
print("x - {}".format(x))
print("type(x) - {}\n".format(type(x)))

print('End dt_author_id.py Python module\n')
