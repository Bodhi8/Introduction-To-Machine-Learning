'''
Created on Mar 6, 2017

@author: Menfi
'''

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
# git add svm_author_id.py
# git svm_author_id.py commit -m "Support VectorMachine Classifier - SVM, C Parameter, C=, kernel='linear', kernel='rbf', .fit, train, accuracy, **!!predict!!**" 
    
import sys
# from time import time
import time

# sys.path.append("../tools/")

# from email_preprocess import preprocess
import email_preprocess

import sklearn.svm 

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = email_preprocess.preprocess()

# only use 1%, 0.01 of features_train - performance increases, time decreases, accuracy decreases
# len(features_train) / 100 
print("len(features_test) - {}".format(len(features_test)))
# print("len(features_train) - {}".format(len(features_train)))
# print("len(features_train) / 100 - {}".format(len(features_train) / 100))
# print("round(len(features_train) / 100) - {}".format(round(len(features_train) / 100)))
# features_train = features_train[:round(len(features_train)/100)] 
# labels_train = labels_train[:round(len(labels_train)/100)]
 
#########################################################
### your code goes here ###

#########################################################

# Create, instantiate, Support Vector Machines - SVM - Classifier
# mySupportVectorMachineClassifier = sklearn.svm.SVC(kernel = 'linear') 
# mySupportVectorMachineClassifier - training time - .fit time - 143.555 using 100% of the data


# Introduction to Machine Learning - SVM - Deploy an RBF Kernel  31 of 37 -  7 March 2017
# Introduction to Machine Learning - SVM - Optimize C Parameter, C=, C Argument  32 of 37 -  7 March 2017
mySupportVectorMachineClassifier = sklearn.svm.SVC(C=10000,kernel = 'rbf') 
# mySupportVectorMachineClassifier - training time - .fit time - 0.104


# This time using Support Vector Machines (SVM) - Support Vector Classifier (SVC)
# Last Time Used Naive Bayes Classifier, Naive Bayes Classifier ran much faster
# Introduction To Machine Learning - Naive Bayes - Machine Learning For Author ID - 40 of 43
# Naive Bayes took seconds, Support Vector Machines (SVM) - Support Vector Classifier (SVC) took minutes
# Naive Bayes Accuracy - 0.973
# Support Vector Machines (SVM) - Support Vector Classifier (SVC) - accuracy - 
# mySupportVectorMachineClassifierScore - 0.9840728100113766 sklearn.svm.SVC(kernel = 'linear')

# mySupportVectorMachineClassifier = sklearn.svm.SVC() # instructor - takes much longer about 15 minutes 
# mySupportVectorMachineClassifier - training time - .fit time - 915.65

# Train (.fit) - Support Vector Machines - SVM - Classifier ** learns patterns ** 
t0 = time.time()
mySupportVectorMachineClassifier.fit(features_train, labels_train)
print("mySupportVectorMachineClassifier - training time - .fit time - {}\n".format(round(time.time() - t0, 3)))
#      mySupportVectorMachineClassifier - training time - .fit time - 143.555 ** sklearn.svm.SVC(kernel = 'linear') **
#      mySupportVectorMachineClassifier - training time - .fit time - 915.65  ** sklearn.svm.SVC() TAKES LONGER **
#      mySupportVectorMachineClassifier - training time - .fit time - 0.104   ** sklearn.svm.SVC(kernel = 'rbf') ** ** using 1% of the data **


t0 = time.time()
mySupportVectorMachineClassifierScore = mySupportVectorMachineClassifier.score(features_test,labels_test)
# print("mySupportVectorMachineClassifierScore - {}\n".format(mySupportVectorMachineClassifierScore))
#      mySupportVectorMachineClassifierScore - 0.9840728100113766 - sklearn.svm.SVC(kernel = 'linear')
#      mySupportVectorMachineClassifierScore - 0.8845278725824801 accuracy dropped when using 1% of features_train and labels_train
#      mySupportVectorMachineClassifierScore - 0.6160409556313993 ** accuracy - sklearn.svm.SVC(kernel = 'rbf') - 0.6160409556313993
print("mySupportVectorMachineClassifierScore - {} ACCURACY\n".format(mySupportVectorMachineClassifierScore))
#      mySupportVectorMachineClassifierScore - 0.8924914675767918 ACCURACY - 1% of data, sklearn.svm.SVC(C=10000,kernel = 'rbf')




print("mySupportVectorMachineClassifier - accuracy time - {}\n".format(round(time.time() - t0, 3)))
#      mySupportVectorMachineClassifier - accuracy time - 16.473

t0 = time.time()
pred = mySupportVectorMachineClassifier.predict(features_test)
print("mySupportVectorMachineClassifier - predict time - {}\n".format(round(time.time() - t0, 3)))

print("pred - {}".format(pred))
print("pred[0] - {}, pred[10] - {}, pred[26] - {}, pred[50] - {}".format(pred[0], pred[10], pred[26], pred[50]))
print("len(pred) - {}".format(len(pred)))
print("type(pred) - {}\n".format(type(pred)))

# sum or count elements in array
num_zeroes = (pred == 0).sum()
print("num_zeroes - {}".format(num_zeroes))

num_ones = (pred == 1).sum()
print("num_ones - {}".format(num_ones))
print("num_ones + num_zeroes - {}".format(num_ones + num_zeroes))








