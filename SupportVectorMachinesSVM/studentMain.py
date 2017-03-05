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
    
# instantiate the Naive Bayes Terrain Classifier
# clf = classify(features_train, labels_train)

# Run Naive Bayes prediction
# pred = clf.predict(features_test) # accuracy methods 2 and 3 below

# 3 methods to get Naive Bayes Terrain Classifier #ACCURACY"

# -----------------------------------------------------

# instantiate the Suport Vector Machines - SVM - Classifier 
# SVMclf  = classify(features_train, labels_train)

# generate Support Vector Machines - SVM - predictor
# SVMpred = SVMclf.predict(features_test)


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
# Instantiate, then get or return the Naive Bayes Gaussain TRAINED classifier
clf = classify('NB', features_train, labels_train)
# print("\ttype(clf) - {}\n".format(type(clf))) # class 'sklearn.naive_bayes.GaussianNB

# Instantiate, then get or return the SupportVectorMachines TRAINED classifier
SVMclf = classify('SVM', features_train, labels_train)

# use the trained classifier to do the predictions
# generate the ** pred ** numpy.ndarray (n dimensional array)
pred = clf.predict(features_test) # accuracy methods 2 and 3 below
# print("\tpred is {}\n".format(pred))
# print("\ttype(pred) - {}\n".format(type(pred))) # numpy.ndarray

# generate Support Vector Machines - SVM - predictor
SVMpred = SVMclf.predict(features_test)

# accuracy of GaussianNB() classifier method 1 of 3
# accuracy - get the accuracy og the classifier # 0.884
my_clf_score = clf.score(features_test,labels_test)
print("\tmy_clf_score is {}".format(my_clf_score)) # 0.884
# print("\ttype(my_clf_score) - {}\n".format(type(my_clf_score))) # class 'numpy.float64'

# apply accuracy method 1 of 3 to the Support Vector Machines - SVM - Terrain Classifier 
my_SVM_clfScore = SVMclf.score(features_test,labels_test)
print("\tmy_SVM_clfScore is {}\n".format(my_SVM_clfScore))

# accuracy of GaussianNB() classifier ACCURACY method 2 of 3
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
        
# accuracy - get the accuracy of the Naive Bayes Classifier # 0.884
print("\tlen(labels_test) is {}".format(len(labels_test))) # 
print("\tNaive Bayes matchCount is {}".format(matchCount)) # 
print("\tNaive Bayes Terrain Classifier Accuracy - matchCount / len(labels_test) is {}".format(matchCount / len(labels_test))) # 0.884

# accuracy of Support Vector Machines - SVM - Terrain Classifier - ACCURACY method 2 of 3
# enumerate through Python list, also keep track of list index 
# enumerate through the labels_test *** list ***
SupportVectorMachinesMatchCount = 0
for idx, label in enumerate(labels_test):
    # print("\tidx is {}".format(idx)) # 0 - 249
    # print("\ttype(idx) - {}\n".format(type(idx))) # int, int
    # print("\tlabel is {}".format(label)) # 0, 1, 1.0
    # print("\ttype(label) - {}\n".format(type(label))) # class 'int', class 'float'
    # print("\tpred[idx] is {}".format(pred[idx])) # 0.0, 1.0 ...
    # print("\ttype(pred[idx]) - {}\n".format(type(pred[idx]))) # numpy.float64
    if label == SVMpred[idx]:
        SupportVectorMachinesMatchCount +=1
        
# accuracy - get the accuracy of the Support Vector Machines - SVM - Terrain Classifier # 
print("\tlen(labels_test) is {}".format(len(labels_test))) # 
print("\tSupportVectorMachinesMatchCount is {}".format(SupportVectorMachinesMatchCount)) # 
print("\tSupport Vector Machine Terrain Classifier Accuracy - SupportVectorMachinesMatchCount / len(labels_test) is {}\n".format(SupportVectorMachinesMatchCount / len(labels_test))) # 

# accuracy of Naive Bayes Terrain Classifier method 3 of 3
myGaussianNBTerrainClassifierAAccuracy = sklearn.metrics.accuracy_score(pred,labels_test)
print("\tmyGaussianNBTerrainClassifierAAccuracy - {}".format(myGaussianNBTerrainClassifierAAccuracy))
# print("\ttype(myGaussianNB_Classifier_Accuracy) - {}\n".format(type(myGaussianNB_Classifier_Accuracy)))

# accuracy of Support Vector Machines - SVM - Terrain Classifier method 3 of 3
SupportVectorMachinesSVMTerraiClassifieAccuracy = sklearn.metrics.accuracy_score(SVMpred,labels_test)
print("\tSupportVectorMachinesSVMTerraiClassifieAccuracy - {}\n".format(SupportVectorMachinesSVMTerraiClassifieAccuracy))
# print("\ttype(SupportVectorMachinesSVMTerraiClassifieAccuracy) - {}\n".format(type(SupportVectorMachinesSVMTerraiClassifieAccuracy)))

### draw Naive Bayes Gaussian Classifier
### draw the decision boundary with the text points overlaid
myPrettyPicture = prettyPicture(clf, features_test, labels_test)

### draw SVM SupportVectorMachines Classifier 
### draw the decision boundary with the text points overlaid
myPrettyPicture = prettyPicture(SVMclf, features_test, labels_test)

# print("\ttype(myPrettyPicture) - {}\n".format(type(myPrettyPicture))) 

# output_image("test.png", "png", open('/Users/Menfi/Documents/workspace/zzzzz/src/test.png', "rb").read())

print('End studentMain.py')


