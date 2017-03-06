'''
Created on Feb 24, 2017

@author: Menfi
'''
# Terrain Classification 

#
# Naive Bayes Classifier
#
# instantiated Naive Bayes Classifier 
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# fit or train the Naive Bayes Classifier
# clf = GaussianNB()

#
# Support Vector Machines - SVM - Classifier
#
# instantiated Support Vector Machines - SVM - Classifier
# import sklearn.svm
# (kernel = 'linear') from instructor 
# SVMclf = sklearn.svm.SVC(kernel = 'linear')

# fit or train the Support Vector Machines - SVM - Classifier
# SVMclf.fit(features_train, labels_train)

# Naive Bayes Classifier 
from sklearn.naive_bayes import GaussianNB

# Support Vector Machines SVM Classifier
import sklearn.svm

import time

def classify(classifierType, features_train, labels_train):
       
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    ### your code goes here!
    
    print('\tBegin ClassifyNB.py - classify function\n')
    # Create, instantiate Naive Bayes Gaussian classifier - clf - classifier
    clf = GaussianNB()
    
    # Create, instantiate Support Vector Machines, SVM, Classifier 
    # SVMclf = sklearn.svm.SVC()
    # instructor saya - kernel = 'linear' 
    # (kernel = 'linear') - visibly STRAIGHTENS out the decision boundary 
    # SVMclf = sklearn.svm.SVC(kernel = 'linear')
    # Experiment with the C parameter
    # result - small C parameter - significantly straighter line
    # SVMclf = sklearn.svm.SVC(C=10) # small C parameter - significantly straighter line
    # kernel : string, optional (default=’rbf’)

    SVMclf = sklearn.svm.SVC(kernel='rbf', gamma=1)
    # SVMclf = sklearn.svm.SVC()
    
    # call the fit function of - *** import the sklearn module for GaussianNB ***
    # call the fit function of the newly created classifier clf 
    # fit or TRAIN the clf classifier, clf, classifier LEARNS PATTERNS 
    # fit or TRAIN, instructor - "give it the TRAINING DATA", instructor - "it LEARNS THE PATTERNS"
    # features_train - Features 
    # labels_train - Labels 
    # clf.fit(features, labels) - generic format
    # print("\tClassifyNB.py - classify function - features_train is {}".format(features_train))
    # print("\tClassifyNB.py - classify function - labels_train is {}".format(labels_train))

    # Train Naive Bayes Gaussian Terrain Classifier 
    t0 = time.time()
    clf.fit(features_train, labels_train)
    print("Naive Bayes, fit / training timing: - {}".format(round(time.time() - t0, 3)))

    print("\ttype(clf) - {}".format(type(clf)))
    print("\tclf.__class__.__name__ - {}\n".format(clf.__class__.__name__))
    
    # Train SupportVectorMachines Classifier 
    t0 = time.time()
    SVMclf.fit(features_train, labels_train)
    print("Support Vector Machines - SVM, fit / training timing: - {}".format(round(time.time() - t0, 3)))

    print("\ttype(SVMclf) - {}".format(type(SVMclf)))
    print("\tSVMclf.__class__.__name__ - {}".format(SVMclf.__class__.__name__))
    print("\ttype(SVMclf.__class__.__name__) - {}\n".format(type(SVMclf.__class__.__name__)))
    
    print('\tENDClassifyNB.py - classify function\n')
    
    if classifierType == 'SVM':
        return SVMclf
    elif classifierType == 'NB':
        return clf
    
'''
    if SVMclf.__class__.__name__ == 'SVC':
        return SVMclf
    elif clf.__class__.__name__ == 'GaussianNB':
        return SVMclf

    # return SVMclf
    # return clf
'''
    
print('Begin ClassifyNB.py')
print('End ClassifyNB.py\n')
    