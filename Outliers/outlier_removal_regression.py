#!/usr/bin/python

from sklearn import linear_model

import random
import numpy
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner

### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "rb") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "rb") )

### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like

reg = linear_model.LinearRegression()
print("\ntype(reg) - {}\n".format(type(reg)))

reg.fit(ages_train, net_worths_train)
print("\ntype(ages_train) - {}".format(type(ages_train)))
print("ages_train.shape - {}".format(ages_train.shape))
print(ages_train[0:2, 0:1]) # get the first two rows, 1 column 

print("\ntype(net_worths_train) - {}".format(type(net_worths_train)))
print("net_worths_train.shape - {}".format(net_worths_train.shape))
print(net_worths_train[0:2, 0:1]) # get the first two rows, 1 column 

print("41\nreg.coef_ - {}\n".format(reg.coef_))

# all previous examples - .score(features_test,labels_test)
# current example -    reg.score(feature_train, target_train)
# reg.score similar to calculating accuracy in Supervised Classifier
# Performance Metrics Used to Evaluate Regressions - score function - the higher the r squared score the better - maximum value - 1
# best practices apply score function to test data not train data - discover overfitting using test data, not train data 
# stats (r squared) on training dataset
#         feature_train - features - inputs - X axis
#                        target_train - outputs - trying to predict - Y axis
reg.score(ages_test, net_worths_test)
print("type(ages_test) - {}".format(type(ages_test)))
print("ages_test.shape - {}".format(ages_test.shape))
print(ages_test[0:2, 0:1]) # get the first two rows, 1 column 

print("\ntype(net_worths_test) - {}".format(type(net_worths_test)))
print("net_worths_test.shape - {}".format(net_worths_test.shape))
print(net_worths_test[0:2, 0:1]) # get the first two rows, 1 column 

print("60reg.score(ages_test, net_worths_test) - regression score - {}\n".format(reg.score(ages_test, net_worths_test)))

try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
    
    # print("type(predictions) - {}\n".format(type(predictions)))
    #        type(predictions) - <class 'numpy.ndarray'>
    print("predictions.shape - {}".format(predictions.shape))
    print(predictions[0:2, 0:1])
    # predictions.shape - (90, 1), [[ 314.65206822], [ 314.65206822]]
    
    # ages_train.shape - (90, 1), [[57], [57]]
    # net_worths_train.shape - (90, 1), [[ 338.08951849], [ 344.21586776]]

except NameError:
    print ("your regression object doesn't exist, or isn't name reg")
    print ("can't make predictions to use in identifying outliers")

### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        print("97\nreg.coef_ - {}\n".format(reg.coef_))
        print("98reg.score(ages_test, net_worths_test) - regression score - {}\n".format(reg.score(ages_test, net_worths_test)))

        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print ("you don't seem to have regression imported/created,")
        print ("   or else your regression object isn't named reg")
        print ("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()

else:
    print ("\noutlierCleaner() is returning an empty list, no refitting to be done")

