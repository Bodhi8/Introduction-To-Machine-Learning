'''
Created on Feb 28, 2017

@author: Menfi
'''

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
    Fresh download - 1 March 2017
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print("len(features_train)- {}".format(len(features_train)))
print("features_train- {}\n".format(features_train))

print("len(features_test)- {}".format(len(features_test)))
print("features_test- {}\n".format(features_test))

print("len(labels_train)- {}".format(len(labels_train)))
print("labels_train- {}\n".format(labels_train))

print("len(labels_test)- {}".format(len(labels_test)))
print("labels_test- {}\n".format(labels_test))

classifier = GaussianNB()
print("classifier- {}".format(classifier))
#      classifier- GaussianNB(priors=None)
print("type(classifier)- {}\n".format(type(classifier)))
#      type(classifier)- <class 'sklearn.naive_bayes.GaussianNB'>

classifier.fit(features_train,labels_train)

pred = classifier.predict(features_test)
print("pred - {}".format(pred))
#      pred - [0 0 1 ..., 1 0 0]
print("type(pred) - {}\n".format(type(pred)))
#      type(pred) - <class 'numpy.ndarray'>

# What is the accuracy of your Naive Bayes author identifier? 1 of 3
my_classifier_score = classifier.score(features_test,labels_test)
print("my_classifier_score - {}".format(my_classifier_score))
#      my_classifier_score - 0.9732650739476678 1 of 3
# print("type(my_classifier_score)- {}\n".format(type(my_classifier_score)))
#        type(my_classifier_score)- <class 'numpy.float64'>

# What is the accuracy of your Naive Bayes author identifier? 2 of 3
matchCount = 0
for idx, label in enumerate(labels_test):
    # print("\tidx is {}".format(idx)) # 0 - 249
    # print("\ttype(idx) - {}\n".format(type(idx))) # int, int
    # print("\tlabel is {}".format(label)) # 0, 1, 1.0
    # print("\ttype(label) - {}\n".format(type(label))) # class 'int', class 'float'
    # print("\tpred[idx] is {}".format(pred[idx])) # 0.0, 1.0 ...
    # print("\ttype(pred[idx]) - {}\n".format(type(pred[idx]))) # numpy.float64
    if label == pred[idx]:
        matchCount +=1
        
print("len(labels_test) is {}".format(len(labels_test))) 
#      len(labels_test) is 1758
print("matchCount is {}".format(matchCount)) 
#      matchCount is 1711
print("matchCount / len(labels_test) is {}\n".format(matchCount / len(labels_test)))
#      matchCount / len(labels_test) is 0.9732650739476678

# What is the accuracy of your Naive Bayes author identifier? 3 of 3
myGaussianNB_Classifier_Accuracy = sklearn.metrics.accuracy_score(pred,labels_test)
print("myGaussianNB_Classifier_Accuracy - {}\n".format(myGaussianNB_Classifier_Accuracy))
#      myGaussianNB_Classifier_Accuracy - 0.9732650739476678
# print("type(myGaussianNB_Classifier_Accuracy) - {}\n".format(type(myGaussianNB_Classifier_Accuracy)))
#      type(myGaussianNB_Classifier_Accuracy) - <class 'numpy.float64'>



#########################################################
### your code goes here ###


#########################################################

print('Begin nb_author_id Python Module')
print('End nb_author_id Python Module')
