'''
Created on Mar 14, 2017

@author: Menfi
'''

from sklearn import linear_model

print('\nBegin linearRegression.py Python module\n')

x = 5
print("x - {}".format(x))
print("type(x) - {}\n".format(type(x)))

myLinearRegressionClassifier = linear_model.LinearRegression()
print("myLinearRegressionClassifier - {}".format(myLinearRegressionClassifier))
print("type(myLinearRegressionClassifier) - {}\n".format(type(myLinearRegressionClassifier)))

myFittedClassifier = myLinearRegressionClassifier.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
# print("myFittedClassifier - {}".format(myFittedClassifier))
#        myFittedClassifier - LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# print("type(myFittedClassifier) - {}\n".format(type(myFittedClassifier)))
#        type(myFittedClassifier) - <class 'sklearn.linear_model.base.LinearRegression'>

# instructor says coefficients or what we call the slope  
myLinearRegressionClassifier.coef_
print("myLinearRegressionClassifier.coef_ - {}".format(myLinearRegressionClassifier.coef_))
#      myLinearRegressionClassifier.coef_ - [ 0.5  0.5]
print("type(myLinearRegressionClassifier.coef_) - {}\n".format(type(myLinearRegressionClassifier.coef_)))

print('End linearRegression.py Python module\n')

