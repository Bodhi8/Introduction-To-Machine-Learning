#!/usr/bin/python

"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# mgc
import sklearn.tree

# from sklearn.model_selection import train_test_split
import sklearn.model_selection

print('\nBegin validate_poy.py Python module\n')

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )
# print("data_dict -> ")
# print(data_dict)
# {'GARLAND C KEVIN': {'
# restricted_stock': 259907, 
# 'expenses': 48405, 
# 'from_messages': 44, 
# 'other': 60814, 
# 'from_this_person_to_poi': 27, 
# 'loan_advances': 'NaN', 
# 'long_term_incentive': 375304, 
# 'email_address': 'kevin.garland@enron.com', 
# 'total_stock_value': 896153, 
# 'total_payments': 1566469, 
# 'director_fees': 'NaN',
 
# 'salary': 231946, # ***
 
# 'bonus': 850000,
 
# 'poi': False, # ***
 
# 'restricted_stock_deferred': 'NaN', 
# 'exercised_stock_options': 636246, 
# 'deferral_payments': 'NaN', 
# 'to_messages': 209, 
# 'from_poi_to_this_person': 10, 
# 'deferred_income': 'NaN', 
# 'shared_receipt_with_poi': 178}
print()
# print("type(data_dict) - {}".format(type(data_dict)))
#        type(data_dict) - <class 'dict'>
# print("len(data_dict) - {}\n".format(len(data_dict)))
#        len(data_dict) - 146



### first element is our labels, poi - labels  
### any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
# print("data -> ")
# print(data)
# poi - salary
# [
# [  0.00000000e+00   2.48146000e+05]
# [  0.00000000e+00   2.59996000e+05]
# print()
# print("type(data) - {}".format(type(data)))
#        type(data) - <class 'numpy.ndarray'>
# print("len(data) - {}\n".format(len(data)))
#      len(data) - 95

labels, features = targetFeatureSplit(data)

# bool poi True - 1 False - 0
# print("labels -> ")
# print(labels)
# [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 
# print()
# print("type(labels) - {}".format(type(labels)))
# type(labels) - <class 'list'>
# print("len(labels) - {}\n".format(len(labels)))
# len(labels) - 95

# print("features -> ")
# print(features)
# [array([ 274975.]), array([ 184899.]), array([ 231946.]), array([ 182245.]), these ARE the salaries
# print()
# print("type(features) - {}".format(type(features)))
# type(features) - <class 'list'>
# print("len(features) - {}\n".format(len(features)))
# len(features) - 95

### it's all yours from here forward!  

myDecisionTreeClassifierInstance = sklearn.tree.DecisionTreeClassifier()

# print("myDecisionTreeClassifierInstance -> ")
# print(myDecisionTreeClassifierInstance)
#  DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')
# print()
# print("type(myDecisionTreeClassifierInstance) - {}".format(type(myDecisionTreeClassifierInstance)))
# type(myDecisionTreeClassifierInstance) - <class 'sklearn.tree.tree.DecisionTreeClassifier'>

# *** previous working example *** sklearn.tree.DecisionTreeClassifier() ***  
# *** .fit(features_train, labels_train), features, then labels *** .fit syntax 
# clf = clf.fit(features_train, labels_train)

features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)
print("type(features_train) - {}".format(type(features_train)))
print("len(features_train) - {}".format(len(features_train)))
# print('features_test ->')
# print(features_test) # salary
# [array([ 259996.]), array([ 182245.]), array([ 76399.]), array([ 262788.]), array([ 365788.]), array([ 339288.]), array([ 236457.]), array([ 261516.]), 
print("type(features_test) - {}".format(type(features_test)))
# Intro To Machine Learning - Evaluation Metrics - Number of People in Test Set  - 29 of 44 - 11 April 2017
# How many people total are in your test set?  - 11 April 2017
print("len(features_test) - {}\n".format(len(features_test)))
#      len(features_test) - 29

# myDecisionTreeClassifierInstance = myDecisionTreeClassifierInstance.fit(features, labels)
myDecisionTreeClassifierInstance = myDecisionTreeClassifierInstance.fit(features_train, labels_train)
# print('features_train')
# print(features_train) # salary
# [array([ 26704229.]), array([ 248546.]), array([ 415189.]), array([ 370448.]), array([ 330546.])

# print('labels_train')
# print(labels_train) # poi
# [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

# print("myDecisionTreeClassifierInstance -> ")
# print(myDecisionTreeClassifierInstance)
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')
# print()
# print("type(myDecisionTreeClassifierInstance) - {}".format(type(myDecisionTreeClassifierInstance)))
# type(myDecisionTreeClassifierInstance) - <class 'sklearn.tree.tree.DecisionTreeClassifier'>

# features_train, features_test, labels_train, labels_test = sklearn.cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)
# features_train, features_test, labels_train, labels_test = sklearn.cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)
# features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)
# ages_train, ages_test, net_worths_train, net_worths_test = sklearn.model_selection.train_test_split(ages, net_worths)
# feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
# features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0) 
# ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)
# features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

print("labels_test -> ")
print(labels_test) # - note 4 ones 1 
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
# print("type(labels_test) - {}".format(type(labels_test)))
#        type(labels_test) - <class 'list'>
# Intro To Machine Learning - Evaluation Metrics - Number of People in Test Set  - 29 of 44 - 11 April 2017
# How many people total are in your test set?  - 11 April 2017
print("len(labels_test) - {}\n".format(len(labels_test)))
#        len(labels_test) - 29

# Intro To Machine Learning - Evaluation Metrics - Accuracy of a Biased Identifier - 30 of 44 - 11 April 2017
# If your identifier predicted 0.0 (not POI) for everyone in the test set, what would your accuracy be?  - 30 of 44 - 11 April 2017
# so force, override to all zeroes, no ones 
# labels_test = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# accuracy - 0.8620689655172413 - correct answer

# sklearn.tree.tree.DecisionTreeClassifier accuracy

accuracy = myDecisionTreeClassifierInstance.score(features_test, labels_test)
print("accuracy - {}\n".format(accuracy))
#      accuracy - 0.7241379310344828
# 11 April 2017 - correct answer - Intro To Machine Learning - Evaluation Metrics - Applying Metrics to Your POI Identifier - 27 of 44 
# 11 April 2017 - Recall that at the end of that project, your identifier had an accuracy (on the test set) of 0.724. 

# How many POIs are in the test set for your POI identifier? - 11 April 2017
# 4 ones - 1 - in pred numpy.ndarray below - therefore 4 POIs
# There are 4 ones - 1 - so 4 pois
pred = myDecisionTreeClassifierInstance.predict(features_test)

print("pred -> ")
print(pred)
print()
# [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0. 0.  1.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
# print("type(pred) - {}".format(type(pred)))
#        type(pred) - <class 'numpy.ndarray'>
# print("len(pred) - {}\n".format(len(pred)))
# len(pred) - 29

'''
Intro To Machine Learning - Evaluation Metrics - Number of True Positives - 31 of 44 - 11 April 2017
Does your identifier have any true positives? - No - Nope  - 31 of 44 - 11 April 2017
# True Positive - actual label AND predicted label - 1 
labels_test - note the location of the ones - 1
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
 pred -  - note the location of the ones - 1
[ 0.   0.   0.   0.   1.   0.   0.   0.   0.   0.   0.   1.   0.   0.   0.   0.   0.   0.   0.   1.   0.   1.   0.   0.   0.   0.   0.   0.   0.]
The ones do not line up, they are not at the same position, index - therefore NO True positives.
'''

# Intro To Machine Learning - Evaluation Metrics - Unpacking Into Precision and Recall - 32 of 44 - 11 April 2017
# What is the precision of your identifier? - 32 of 44 - 11 April 2017

from sklearn.metrics import precision_score

# precision_score(y_true, y_pred, labels, pos_label, average, sample_weight) - cut, paste documentation 
myPrecisionScore = precision_score(labels_test, pred) # my precision_score Python code 

print("myPrecisionScore -> ")
print(myPrecisionScore)
# 0.0 - right answer - 11 April 2017
# see above - The ones do not line up, they are not at the same position, index - therefore NO True positives.
# print("type(myPrecisionScore) - {}".format(type(myPrecisionScore)))
#        type(myPrecisionScore) - <class 'numpy.float64'>
print()


# Intro To Machine Learning - Evaluation Metrics - Recall of Your POI Identifier - 33 of 44 - 11 April 2017
# What's the recall of your POI identifier? - 33 of 44 - 11 April 2017
from sklearn.metrics import recall_score
# recall_score(y_true, y_pred, labels, pos_label, average, sample_weight) - cut, paste documentation
myRecallScore = recall_score(labels_test, pred)
print("myRecallScore -> ")
print(myRecallScore)
print()
print("type(myRecallScore) - {}".format(type(myRecallScore)))



# print("type(myPrecisionScore) - {}".format(type(myPrecisionScore)))


print('\nEnd validate_poy.py Python module\n')

