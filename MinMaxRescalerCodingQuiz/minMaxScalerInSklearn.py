'''
Created on Mar 24, 2017

@author: Menfi
'''
import sklearn
import sklearn.preprocessing.data
import numpy
from sklearn.preprocessing.data import MinMaxScaler


# Each element of the numpy array is a different training point
# Each element within the training point is a feature 
# This example one feature - the weights feature 
# Three different training points 
# old_weights = numpy.array([[115], [140], [175]])
weights = numpy.array([[115.], [140.], [175.]])
# print("type(weights) - {}\n".format(type(weights)))
#        type(weights) - <class 'numpy.ndarray'>

scaler = MinMaxScaler()
# print("type(scaler) - {}\n".format(type(scaler)))
#        type(scaler) - <class 'sklearn.preprocessing.data.MinMaxScaler'>

rescaled_weight = scaler.fit_transform(weights)
# 1 of 2 steps fit - find x_min, x_max 
# 2 of 2 steps transform - applies the formula to the elements

# print("rescaled_weight - ")
# print(rescaled_weight)
# rescaled_weight - 
# [[ 0.        ]
#  [ 0.41666667]
#  [ 1.        ]]

print("type(rescaled_weight) - {}\n".format(type(rescaled_weight)))
#      type(rescaled_weight) - <class 'numpy.ndarray'>






