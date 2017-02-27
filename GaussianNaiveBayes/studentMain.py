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

print('Begin studentMain.py')

# prep_terrain_data.py - makeTerrainData() 
features_train, labels_train, features_test, labels_test = makeTerrainData()
print("\tfeatures_train is {}".format(features_train))
print("\tlabels_train is {}".format(labels_train))
print("\tfeatures_test is {}".format(features_test))
print("\tlabels_test is {}\n".format(labels_test))

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.
clf = classify(features_train, labels_train)


### draw the decision boundary with the text points overlaid
myPrettyPicture = prettyPicture(clf, features_test, labels_test)
# print("\ttype(myPrettyPicture) - {}\n".format(type(myPrettyPicture))) 

# output_image("test.png", "png", open('/Users/Menfi/Documents/workspace/zzzzz/src/test.png', "rb").read())

print('End studentMain.py')


