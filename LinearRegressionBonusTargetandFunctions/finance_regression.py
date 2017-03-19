#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    

import sys
import pickle
import itertools

# sys.path.append("../tools/")
# from feature_format import featureFormat, targetFeatureSplit
from tools import feature_format

from sklearn import linear_model
 
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "rb") )

# print("\ntype(dictionary) - {}\n".format(type(dictionary)))
#          type(dictionary) - <class 'dict'> 

print("\nlist(itertools.islice(dictionary.items(), 0, 1)) - {}\n".format(list(itertools.islice(dictionary.items(), 0, 1))))
# list(itertools.islice(dictionary.items(), 0, 1)) - [('PIRO JIM', {
#     'loan_advances': 'NaN', 'from_poi_to_this_person': 0, 'director_fees': 'NaN', 'long_term_incentive': 'NaN', 'salary': 'NaN',
#     'from_this_person_to_poi': 1, 'total_stock_value': 47304, 'email_address': 'jim.piro@enron.com', 'to_messages': 58, 'from_messages': 16,
#     'restricted_stock_deferred': 'NaN', 'poi': False, 'deferral_payments': 'NaN', 'bonus': 'NaN', 'shared_receipt_with_poi': 3,
#     'restricted_stock': 47304, 'expenses': 'NaN', 'deferred_income': 'NaN', 'other': 'NaN', 'exercised_stock_options': 'NaN', 'total_payments': 'NaN'})]

# **'long_term_incentive': 'NaN', **

# print("type(itertools.islice(dictionary.items(), 0, 1)) - {}".format(type(itertools.islice(dictionary.items(), 0, 1))))
#        type(itertools.islice(dictionary.items(), 0, 1)) - <class 'itertools.islice'>

# cast to a list 
print("type(list(itertools.islice(dictionary.items(), 0, 1))) - {}".format(type(list(itertools.islice(dictionary.items(), 0, 1)))))
#      type(list(itertools.islice(dictionary.items(), 0, 1))) - <class 'list'>

### list the features you want to look at--first item in the 
### list will be the "target" feature
# investigate - how closely related is "salary" to bonus
# bonus - target feature
# salary used to predict target - bonus
features_list = ["bonus", "salary"] # dictionary keys - comment uncomment here
#reg.score(feature_test, target_test) - r-squared score - test dataset - -1.48499241736851
# lower score - salary worse at predicting bonus

# investigate - how closely related is "long_term_incentive" to bonus
# features_list = ["bonus", "long_term_incentive"]  # - comment uncomment here
#reg.score(feature_test, target_test) - r-squared score - test dataset - -0.5927128999498643
# higher score - 'long_term_incentive' better at predicting bonus

# data = feature_format.featureFormat( dictionary, features_list, remove_any_zeroes=True)
data = feature_format.featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = 'tools/python2_lesson06_keys.pkl')
# print("type(data) - {}\n".format(type(data)))
#      type(data) - <class 'numpy.ndarray'>

# $ cat python2_lesson06_keys.pkl 
# (lp0
# S'METTS MARK'
# p1
# aS'BAXTER JOHN C'
# p2

print("data.shape - {}\n".format(data.shape))
# data.shape - (79, 2) 79 rows two columns

print("data[0:3] -> ") # features_list = ["bonus", "salary"] not features_list = ["bonus", "long_term_incentive"]
print(data[0:3])
# [[  600000.   365788.]
#  [ 1200000.   267102.]
#  [  350000.   170941.]

# print(data[0:2]) # first two rows. two columns 
# print(data[0:2,0]) # first two rows, column 1 of 2 only - zero based indexing 
# print(data[0:2,1]) # first two rows, column 2 of 2 only - zero based indexing 
# print(data[0:2,0:1]) # first two rows, column 1 of 1 only - zero based indexing
# print(data[0:2,0:2]) # first two rows, two columns - zero based indexing - this is a good example 

# target, features = targetFeatureSplit( data )
target, features = feature_format.targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.

reg = linear_model.LinearRegression()
# print("reg - {}".format(reg))
#      reg - LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# print("type(reg) - {}".format(type(reg)))
#        type(reg) - <class 'sklearn.linear_model.base.LinearRegression'>

reg.fit (feature_train, target_train)

print("\nreg.coef_ 'slope' - {}".format(reg.coef_))
#      reg.coef_ 'slope' - [ 5.44814029]

print("reg.intercept_ - {} (Y intercept)\n".format(reg.intercept_))
#      reg.intercept_ - -102360.54329387983 (Y intercept)

print("feature_train[0 : 2] - ")
print(feature_train[0 : 2])
# [array([ 415189.]), array([ 243293.])]
# print("type(feature_train) - {}".format(type(feature_train)))
#        type(feature_train) - <class 'list'>

print("target_train[0 : 2] - ")
print(target_train[0 : 2])
# [1000000.0, 1500000.0]
# print("type(target_train) - {}\n".format(type(target_train)))
#        type(target_train) - <class 'list'>

print("\nfeature_test[0 : 2] - ")
print(feature_test[0 : 2])

print("target_test[0 : 2] - ")
print(target_test[0 : 2])

# all previous examples - .score(features_test,labels_test)
# current example -    reg.score(feature_train, target_train)
# reg.score similar to calculating accuracy in Supervised Classifier
# Performance Metrics Used to Evaluate Regressions - score function - the higher the r squared score the better - maximum value - 1
# best practices apply score function to test data not train data - discover overfitting using test data, not train data 
# stats (r squared) on training dataset
#         feature_train - features - inputs - X axis
#                        target_train - outputs - trying to predict - Y axis
reg.score(feature_train, target_train)
print("\nreg.score(feature_train, target_train) - r-squared score - training dataset - {}".format(reg.score(feature_train, target_train)))
#      reg.score(feature_train, target_train) - r-squared score - training dataset - 0.04550919269952436
# instructor - 0.046

# all previous examples - .score(features_test,labels_test)
# current example -    reg.score(feature_test, target_test)
# reg.score similar to calculating accuracy in Supervised Classifier
# Performance Metrics Used to Evaluate Regressions - score function - the higher the r squared score the better - maximum value - 1
# best practices apply score function to test data not train data - discover overfitting using test data, not train data 
# stats (r squared) on training dataset
#         feature_test - features - inputs - X axis
#                       target_test - outputs - trying to predict - Y axis
# reg.score(feature_test, target_test)
print("reg.score(feature_test, target_test) - r-squared score - test dataset - {}\n".format(reg.score(feature_test, target_test)))
# reg.score(feature_test, target_test) - r-squared score - test dataset - -1.48499241736851
# lower score - salary worse at predicting bonus

# investigate - how closely related is "salary" to bonus
# features_list = ["bonus", "salary"] # dictionary keys
# instructor We got -1.485 -- pretty bad score, huh?
# reg.score(feature_test, target_test) - r-squared score - test dataset - -1.48499241736851
# lower score - salary worse at predicting bonus

# investigate - how closely related is "long_term_incentive" to bonus
# features_list = ["bonus", "long_term_incentive"]
# instructor - 
# reg.score(feature_test, target_test) - r-squared score - test dataset - -0.5927128999498643
# higher score - 'long_term_incentive' better at predicting bonus

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt

# edit change the figure size, 8 wider, 7 taller 
plt.figure(figsize=(8,7))

for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass

reg.fit(feature_test, target_test)
print("\nreg.coef_ 'slope' - {}".format(reg.coef_))

plt.plot(feature_train, reg.predict(feature_train), color="b") 

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
