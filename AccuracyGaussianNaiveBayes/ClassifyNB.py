'''     
Created on Feb 24, 2017

@author: Menfi
'''

from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):
       
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    ### your code goes here!
    
    print('\tBegin ClassifyNB.py - classify function')
    # Create classifier - clf - classifier
    clf = GaussianNB()
    
    # call the fit function of - *** import the sklearn module for GaussianNB ***
    # call the fit function of the newly created classifier clf 
    # fit or TRAIN the clf classifier, clf, classifier LEARNS PATTERNS 
    # fit or TRAIN, instructor - "give it the TRAINING DATA", instructor - "it LEARNS THE PATTERNS"
    # features_train - Features 
    # labels_train - Labels 
    # clf.fit(features, labels) - generic format
    # print("\tClassifyNB.py - classify function - features_train is {}".format(features_train))
    # print("\tClassifyNB.py - classify function - labels_train is {}".format(labels_train))

    clf.fit(features_train, labels_train)

    print('\tENDClassifyNB.py - classify function\n')
    return clf
    
print('Begin ClassifyNB.py')
# classify('a', 'b')
print('End ClassifyNB.py\n')
    
