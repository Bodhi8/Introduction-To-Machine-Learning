'''
Created on Mar 14, 2017

@author: Menfi
'''

import matplotlib.pyplot as plt
from studentRegression import studentReg
from class_vis import prettyPicture, output_image

from ages_net_worths import ageNetWorthData

print('\nBegin studentMain.py\n')

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

print("\nages_train[0:2,0:3] - {}".format(ages_train[0:2,0:3]))
print("\nages_train[0:2] - ")
print(ages_train[0:2,])
# [[22] [63]]
#print("type(ages_train) - {}".format(type(ages_train)))
#       type(ages_train) - <class 'numpy.ndarray'>
# reg.fit (ages_train, net_worths_train) - studentRegression.py
# historically - .fit(features_train,labels_train) - ages_train - features_train - X axis 

print("\nnet_worths_train[0:2,0:3] - {}".format(net_worths_train[0:2,0:3]))
print("\nnet_worths_train[0:2] - ")
print(net_worths_train[0:2,])
# [[  66.97839379] [ 373.01919127]]
# print("type(net_worths_train) - {}\n".format(type(net_worths_train)))
#        type(net_worths_train) - <class 'numpy.ndarray'>
# reg.fit (ages_train, net_worths_train) - studentRegression.py
# historically - .fit(features_train,labels_train) - net_worths_train - labels_train - Y axis 

# Call the studentRegression.py function studentReg passing ages_train, net_worths_train
reg = studentReg(ages_train, net_worths_train)

print("reg.coef_ 'slope' - {}\n".format(reg.coef_))
#      reg.coef_ - [[ 6.30945055]] - slope
# print("type(reg.coef_) - {}\n".format(type(reg.coef_)))
#       type(reg.coef_) - <class 'numpy.ndarray'> - slope

# predict net worth @ age 27, 37 
print("reg.predict([27]) - {}".format(reg.predict(27)))
#      reg.predict([27]) - [[ 162.90800279]]
print("reg.predict([37]) - {}\n".format(reg.predict(37)))
#      reg.predict([37]) - [[ 226.00250833]]
# print("type(reg.predict([37])) - {}\n".format(type(reg.predict(37))))
#      type(reg.predict([37])) - <class 'numpy.ndarray'>

# y intercept of the DATA, NOT the black line - plt.plot(ages_test, reg.predict(ages_test), color="black")
print("reg.intercept_ - {}\n".format(reg.intercept_))
#      reg.intercept_ - [-7.44716216]

# all previous examples - .score(features_test,labels_test)
# current example -    reg.score(ages_test,   net_worths_test)
# reg.score similar to calculating accuracy in Supervised Classifier
# Performance Metrics Used to Evaluate Regressions - score function - the higher the r squared score the better - maximum value - 1
# best practices apply score function to test data not train data - discover overfitting using test data, not train data 
# stats (r squared) on test dataset
#         ages_test - features - inputs - X axis
#                     net_worths_test - outputs - trying to predict - Y axis
reg.score(ages_test, net_worths_test)
print("reg.score(ages_test, net_worths_test) - r-squared score - test dataset - {}".format(reg.score(ages_test, net_worths_test)))

# all previous examples - .score(features_test,labels_test)
# current example -    reg.score(ages_train,   net_worths_train)
# reg.score similar to calculating accuracy in Supervised Classifier
# Performance Metrics Used to Evaluate Regressions - score function - the higher the r squared score the better - maximum value - 1
# best practices apply score function to test data not train data - discover overfitting using test data, not train data 
# stats (r squared) on training dataset
#         ages_train - features - inputs - X axis
#                     net_worths_train - outputs - trying to predict - Y axis
reg.score(ages_train, net_worths_train)
print("reg.score(ages_train, net_worths_train) - r-squared score - training dataset - {}\n".format(reg.score(ages_train, net_worths_train)))

plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
# visualize regression on top of scatter points - the black line
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2) # location of legend box see udacity legal size  folder
plt.xlabel("ages")
plt.ylabel("net worths")
plt.show()

plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())

print('End studentMain.py\n')
