'''
Created on Mar 8, 2017

@author: Menfi

git add studentMain.py
git commit studentMain.py -m "udacity.com - Introduction To Machine Learning Examples - ** Decision Tree ** -  Supervised Classification Algorithm"
'''


print('\nBegin studentMain.py Python module\n')

x = 5
print("x - {}\n".format(x))
print("type(x) - {}\n".format(type(x)))


""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()
print("type(features_train) - {}".format(type(features_train)))
print("type(labels_train) - {}".format(type(labels_train)))
print("type(features_test) - {}".format(type(features_test)))
print("type(labels_test) - {}\n".format(type(labels_test)))

### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)

#### grader code, do not modify below this line
prettyPicture(clf, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())

print('\nEnd studentMain.py Python module\n')

