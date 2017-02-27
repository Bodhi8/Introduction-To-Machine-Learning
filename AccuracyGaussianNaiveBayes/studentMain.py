'''
Created on Feb 24, 2017

@author: Menfi
'''
#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify
# import ClassifyNB

import numpy as np
import pylab as pl
import sklearn.metrics

print('Begin studentMain.py')

# prep_terrain_data.py - makeTerrainData() 
features_train, labels_train, features_test, labels_test = makeTerrainData()
print("\tfeatures_train is {}".format(features_train))
print("\tlabels_train is {}".format(labels_train))
print("\tfeatures_test is {}\n".format(features_test))
print("\tlabels_test is {}".format(labels_test))
print("\ttype(labels_test) is {}\n".format(type(labels_test)))

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# You will need to complete this function imported from the ClassifyNB script. 
# Be sure to change to that code tab to complete this quiz.
# instantiate then get or return the TRAINED classifier
clf = classify(features_train, labels_train)
# print("\ttype(clf) - {}\n".format(type(clf))) # class 'sklearn.naive_bayes.GaussianNB

# use the trained classifier to do the predictions
# generate the ** pred ** numpy.ndarray (n dimensional array)
pred = clf.predict(features_test) # accuracy methods 2 and 3 below
# print("\tpred is {}\n".format(pred))
# print("\ttype(pred) - {}\n".format(type(pred))) # numpy.ndarray

# accuracy of GaussianNB() classifier method 1 of 3
# accuracy - get the accuracy og the classifier # 0.884
my_clf_score = clf.score(features_test,labels_test)
print("\tmy_clf_score is {}\n".format(my_clf_score)) # 0.884
# print("\ttype(my_clf_score) - {}\n".format(type(my_clf_score))) # class 'numpy.float64'

# accuracy of GaussianNB() classifier method 2 of 3
# enumerate through Python list, also keep track of list index 
# enumerate through the labels_test *** list ***
matchCount = 0
for idx, label in enumerate(labels_test):
    # print("\tidx is {}".format(idx)) # 0 - 249
    # print("\ttype(idx) - {}\n".format(type(idx))) # int, int
    # print("\tlabel is {}".format(label)) # 0, 1, 1.0
    # print("\ttype(label) - {}\n".format(type(label))) # class 'int', class 'float'
    # print("\tpred[idx] is {}".format(pred[idx])) # 0.0, 1.0 ...
    # print("\ttype(pred[idx]) - {}\n".format(type(pred[idx]))) # numpy.float64
    if label == pred[idx]:
        matchCount +=1
        # print('we have a winner')
        
# accuracy - get the accuracy of the classifier # 0.884
print("\tlen(labels_test) is {}".format(len(labels_test))) # 
print("\tmatchCount is {}".format(matchCount)) # 
print("\tmatchCount / len(labels_test) is {}\n".format(matchCount / len(labels_test))) # 0.884

# accuracy of GaussianNB() classifier method 3 of 3
myGaussianNB_Classifier_Accuracy = sklearn.metrics.accuracy_score(pred,labels_test)
print("\tmyGaussianNB_Classifier_Accuracy - {}".format(myGaussianNB_Classifier_Accuracy))
print("\ttype(myGaussianNB_Classifier_Accuracy) - {}\n".format(type(myGaussianNB_Classifier_Accuracy)))

### draw the decision boundary with the text points overlaid
myPrettyPicture = prettyPicture(clf, features_test, labels_test)
# print("\ttype(myPrettyPicture) - {}\n".format(type(myPrettyPicture))) 

# output_image("test.png", "png", open('/Users/Menfi/Documents/workspace/zzzzz/src/test.png', "rb").read())

print('End studentMain.py')


