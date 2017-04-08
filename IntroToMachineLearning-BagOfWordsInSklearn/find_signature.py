#!/usr/bin/python

import pickle
import numpy
import time
# from vectorize_text import word_data
numpy.random.seed(42)

print('\nBegin find_signature.py Python module\n')

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
# words_file = "../text_learning/your_word_data.pkl" 

# simply a string 
words_file = "your_word_data.pkl" 
# print("words_file - {}".format(words_file))
#      words_file - your_word_data.pkl   
# print("type(words_file) - {}".format(type(words_file)))
#        type(words_file) - <class 'str'>
# print("len(words_file) - {}".format(len(words_file)))
#        len(words_file) - 18

# use the string to open the file,  open your_word_data.pkl
word_data = pickle.load( open(words_file, "rb"))
# print("word_data - {}".format(word_data)) # too long
#        word_data - ['sbaile2 nonprivilegedpst susan pleas send the forego list to rich AOK
# print("type(word_data) - {}".format(type(word_data)))
#        type(word_data) - <class 'list'>
# print("len(word_data) - {}".format(len(word_data)))
#      len(word_data) - 17578

# print("word_data[0] - {}\n".format(word_data[0]))
#        word_data[0] - sbaile2 nonprivilegedpst susan pleas send the forego list to AOK
# print("type(word_data[0]) - {}\n".format(type(word_data[0])))
#        type(word_data[0]) - <class 'str'>
# print("len(word_data[0]) - {}\n".format(len(word_data[0])))
#        len(word_data[0]) - 173

# simply a string 
authors_file = "your_email_authors.pkl"
# print("type(words_file) - {}\n".format(type(words_file)))
#        type(words_file) - <class 'str'>
# print("len(words_file) - {}\n".format(len(words_file)))
#        len(words_file) - 18

# use the string to open the file, open your_word_data.pkl
authors = pickle.load( open(authors_file, "rb") )
# print("authors - {}".format(authors)) # too long
#      authors - [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
# print("type(authors) - {}".format(type(authors)))
#        type(authors) - <class 'list'>
# print("len(authors) - {}".format(len(authors)))
#      len(authors) - 17578
# print("authors[28] - {}".format(authors[28]))
#      authors[28] - 0
# print("type(authors[28]) - {}".format(type(authors[28])))
#        type(authors[28]) - <class 'int'>
# print("authors[17000] - {}\n".format(authors[17000]))
#      authors[17000] - 1
# print("type(authors[1700]) - {}\n".format(type(authors[1700])))

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
# from sklearn import cross_validation
from sklearn import model_selection
# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)
# labels_test # USED HERE myDecisionTreeClassifierAccuracy = myDecisionTreeClassifier.score(features_test,labels_test)

# print('features_train information')
# print("features_train - {}".format(features_train)) # too long
# features_train - ['sshacklensf kay ill give you my comm
# print("type(features_train) - {}".format(type(features_train)))
#        type(features_train) - <class 'list'>
# print("len(features_train) - {}".format(len(features_train)))
#        len(features_train) - 15820
# print("features_train[0] - {}\n".format(features_train[0]))
#      features_train[0] - sshacklensf kay ill give you my comment to the te  
# print("type(features_train[0]) - {}".format(type(features_train[0])))
#        type(features_train[0]) - <class 'str'>
# print("len(features_train[0]) - {}\n".format(len(features_train[0])))
#      len(features_train[0]) - 606

# print('features_test information')
# print("features_test - {}".format(features_test))
#      features_test - ['sshacklensf melissa i went back to your origin d
# print("type(features_test) - {}".format(type(features_test)))
#        type(features_test) - <class 'list'>
# print("len(features_test) - {}".format(len(features_test)))
#      len(features_test) - 1758
# print("features_test[0] - {}\n".format(features_test[0]))
#      features_test[0] - sshacklensf melissa i went back to your origin
# print("type(features_test[0]) - {}".format(type(features_test[0])))
#       type(features_test[0]) - <class 'str'>
# print("len(features_test[0]) - {}\n".format(len(features_test[0])))
# len(features_test[0]) - 329

# print('labels_train information')
# print("labels_train - {}".format(labels_train))
#        labels_train - [0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
# print("type(labels_train) - {}".format(type(labels_train)))
#        type(labels_train) - <class 'list'>
# print("len(labels_train) - {}".format(len(labels_train)))
#        len(labels_train) - 15820
# print("labels_train[0] - {}\n".format(labels_train[0]))
# labels_train[0] - 0
# print("type(labels_train[0]) - {}\n".format(type(labels_train[0])))
#        type(labels_train[0]) - <class 'int'>

# print('124labels_test information')
# print(labels_test)
# [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
# print("type(labels_test) - {}".format(type(labels_test)))
#        type(labels_test) - <class 'list'>
# print("len(labels_test) - {}".format(len(labels_test)))
#      len(labels_test) - 1758
# print("labels_test[0] - {}\n".format(labels_test[0]))
#      labels_test[0] - 0
# print("type(labels_test[0]) - {}\n".format(type(labels_test[0])))
#      type(labels_test[0]) - <class 'int'>

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')

features_train = vectorizer.fit_transform(features_train)
vectorizer.get_feature_names()
# print("vectorizer.get_feature_names() - {}".format(vectorizer.get_feature_names()))
#        vectorizer.get_feature_names() - ['00', '000', '0000', '00000', '0000000', '00000000', '000000000083213', '00000365797'
# print("type(vectorizer.get_feature_names()) - {}".format(type(vectorizer.get_feature_names())))
#        type(vectorizer.get_feature_names()) - <class 'list'>
# print("len(vectorizer.get_feature_names()) - {}".format(len(vectorizer.get_feature_names())))
# len(vectorizer.get_feature_names()) - 37863
print("vectorizer.get_feature_names()[33614] - {}".format(vectorizer.get_feature_names()[33614]))
#      vectorizer.get_feature_names()[33614] - sshacklensf

print("vectorizer.get_feature_names()[14343] - {}".format(vectorizer.get_feature_names()[14343]))
#      vectorizer.get_feature_names()[14343] - cgermannsf
# Sebastian - Yes, this is the next most "highly powerful" word.

print("vectorizer.get_feature_names()[21323] - {}".format(vectorizer.get_feature_names()[21323]))
#      vectorizer.get_feature_names()[21323] - houectect

# print("type(vectorizer.get_feature_names()[33614]) - {}".format(type(vectorizer.get_feature_names()[33614])))
#        type(vectorizer.get_feature_names()[33614]) - <class 'str'>
# print("len(vectorizer.get_feature_names()[33614]) - {}\n".format(len(vectorizer.get_feature_names()[33614])))
#        len(vectorizer.get_feature_names()[33614]) - 11


# print('vectorizer features_train information')
# print("features_train")
# print(features_train)
# time.sleep(3)
# (0, 33614)    0.0480462971635
# (0, 23513)    0.115963746479
# print("features_train.get_shape() - {}".format(features_train.get_shape()))
#      features_train.get_shape() - (15820, 37863)
# print("features_train.getformat() - {}".format(features_train.getformat()))
#      features_train.getformat() - csr
# print("features_train.getH() - {}".format(features_train.getH()))
#      features_train.getH() -   (33614, 0)    0.0480462971635
# print("type(features_train) - {}".format(type(features_train)))
#        type(features_train) - <class 'scipy.sparse.csr.csr_matrix'>

#      type(features_train) - <class 'scipy.sparse.csr.csr_matrix'>
# print("features_train.getnnz() - {}".format(features_train.getnnz()))
#        features_train.getnnz() - 950025

# print("features_train - ")
# print(features_train)
# (0, 33614)    0.0480462971635
# (0, 23513)    0.115963746479

features_test  = vectorizer.transform(features_test).toarray() # USED HERE myDecisionTreeClassifierAccuracy = myDecisionTreeClassifier.score(features_test,labels_test)
# print('vectorizer features_test information')
# print(features_test)
# [
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]
# print()
# print("type(features_test) - {}".format(type(features_test)))
#        type(features_test) - <class 'numpy.ndarray'>
# print("len(features_test) - {}".format(len(features_test)))
#        len(features_test) - 1758
# print("features_test[0] - {}".format(features_test[0]))
#        features_test[0] - [ 0.  0.  0. ...,  0.  0.  0.]
# print("type(features_test[0]) - {}".format(type(features_test[0])))
#        type(features_test[0]) - <class 'numpy.ndarray'>
# print("len(features_test[0]) - {}".format(len(features_test[0])))
#        len(features_test[0]) - 37863
# print("features_test[0][0] - {}\n".format(features_test[0][0]))
#      features_test[0][0] - 0.0
# print("type(features_test[0][0]) - {}\n".format(type(features_test[0][0])))
#        type(features_test[0][0]) - <class 'numpy.float64'>

# Sebastian - Yup! We've limited our training data quite a bit so we should be expecting our models to potentially overfit.
### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray() # USED HERE myDecisionTreeClassifier.fit(features_train,labels_train)
# print("features_train - ")# 
# print(features_train)
# [
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]
# print("type(features_train) - {}".format(type(features_train)))
#        type(features_train) - <class 'numpy.ndarray'>
# print("len(features_train) - {}".format(len(features_train)))
#        len(features_train) - 150
# print("features_train[0] - ")
#  print(features_train[0])
# features_train[0] - [ 0.  0.  0. ...,  0.  0.  0.]
# print("type(features_train[0]) - {}".format(type(features_train[0])))
#        type(features_train[0]) - <class 'numpy.ndarray'>
# print("len(features_train[0]) - {}".format(len(features_train[0])))
#        len(features_train[0]) - 37863
# print("features_train[0][0] - {}\n".format(features_train[0][0]))
# print("type(features_train[0][0]) - {}\n".format(type(features_train[0][0])))
#        type(features_train[0][0]) - <class 'numpy.float64'>


# Sebastian - Yup! We've limited our training data quite a bit so we should be expecting our models to potentially overfit.
labels_train   = labels_train[:150]             # USED HERE myDecisionTreeClassifier.fit(features_train,labels_train)
# print("labels_train - ") 
# print(labels_train)
# [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1
# print("type(labels_train) - {}".format(type(labels_train)))
#        type(labesl_train) - <class 'list'>
# print("len(labesl_train) - {}".format(len(labels_train)))
#        len(labels_train) - 150
# print("labels_train[0] - {}".format(labels_train[0]))
#        labels_train[0] - 0
# print("type(labesl_train[0]) - {}\n".format(type(labels_train[0])))
#        type(labesl_train[0]) - <class 'int'>


# myDecisionTreeClassifierAccuracy - 0.9476678043230944 *** Note 0.947 ***
# Sebastian - Yes, the test performance has an accuracy much higher than it is expected to be - 
# if we are overfitting, then the test performance should be relatively low.
### your code goes here
from sklearn import tree
myDecisionTreeClassifier = tree.DecisionTreeClassifier()
myDecisionTreeClassifier.fit(features_train,labels_train)
myDecisionTreeClassifierAccuracy = myDecisionTreeClassifier.score(features_test,labels_test)
print("myDecisionTreeClassifierAccuracy - {}\n".format(myDecisionTreeClassifierAccuracy))

# print("myDecisionTreeClassifier.feature_importances_ - {}".format(myDecisionTreeClassifier.feature_importances_))
#        myDecisionTreeClassifier.feature_importances_ - [ 0.  0.  0. ...,  0.  0.  0.]
# print("type(myDecisionTreeClassifier.feature_importances_) - {}".format(type(myDecisionTreeClassifier.feature_importances_)))
#        type(myDecisionTreeClassifier.feature_importances_) - <class 'numpy.ndarray'>
# print("len(myDecisionTreeClassifier.feature_importances_) - {}".format(len(myDecisionTreeClassifier.feature_importances_)))
# len(myDecisionTreeClassifier.feature_importances_) - 37863
# print("myDecisionTreeClassifier.feature_importances_[0] - {}".format(myDecisionTreeClassifier.feature_importances_[0]))
#        myDecisionTreeClassifier.feature_importances_[0] - 0.0
# print("type(myDecisionTreeClassifier.feature_importances_[0]) - {}\n".format(type(myDecisionTreeClassifier.feature_importances_[0])))
#        type(myDecisionTreeClassifier.feature_importances_[0]) - <class 'numpy.float64'>


mySaveIndex = 0
myHighCounter = 0
most_important_feature_importance = 0
for feature_importance in myDecisionTreeClassifier.feature_importances_:
    if feature_importance > most_important_feature_importance:
        most_important_feature_importance = feature_importance
 
      
#
# This is where we identify myDecisionTreeClassifier.feature_importances_ and the index of that value
# that index is used above to look up the text, word, associated with the high myDecisionTreeClassifier.feature_importances_ of interest
#  
# print("mySaveFloat - {}".format(mySaveFloat))
#      mySaveFloat - 0.7647058823529412

# loop through the myDecisionTreeClassifier.feature_importances_ class 'numpy.ndarray
# look for and save the largest float, keep track of the index too 
# this is where we identify the index and large myDecisionTreeClassifier.feature_importances_ of interest
# next the index will be used ABOVE to finally get the txet, word associated with the high myDecisionTreeClassifier.feature_importances_
# print("vectorizer.get_feature_names()[33614] - {}".format(vectorizer.get_feature_names()[33614]))
#      vectorizer.get_feature_names()[33614] - sshacklensf

most_important_feature_importance = 0
for index, feature_importance in numpy.ndenumerate(myDecisionTreeClassifier.feature_importances_):
    # print("{} - {}".format(index, x))
    if feature_importance > most_important_feature_importance:
        most_important_feature_importance = feature_importance
        mySaveIndex = index 
        
print("mySaveIndex - {}".format(mySaveIndex))
#      mySaveIndex - (33614,)
print("mySaveIndex[0] - {}".format(mySaveIndex[0]))
#      mySaveIndex[0] - 33614

# next above we use the index to finally find the word or text of interest
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
#                              stop_words='english')
# 
# features_train = vectorizer.fit_transform(features_train)
# vectorizer.get_feature_names()
# print("vectorizer.get_feature_names()[33614] - {}".format(vectorizer.get_feature_names()[33614]))

print("most_important_feature_importance - {}".format(most_important_feature_importance))
# mySaveFloat - 0.7647058823529412 - Right Answer

# right answer using udacity terminology
print("index or number of most important feature - {} -- most important feature value - {}\n".format(mySaveIndex[0], most_important_feature_importance))
#      index or number of most important feature - 33614 -- most important feature value - 0.7647058823529412


# In order to figure out what words are causing the problem,
# you need to go back to the TfIdf and use the feature numbers that you obtained
# in the previous part of the mini-project to get the associated words.
# You can return a list of all the words in the TfIdf by calling get_feature_names() on it;
# pull out the word that’s causing most of the discrimination of the decision tree.
# What is it? Does it make sense as a word that’s uniquely tied to either Chris Germany or Sara Shackleton, a signature of sorts?

# *** this follow up work is done right here in find_signature.py ***
# beginning here - see TfidfVectorizer code above *** see above *** 
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
# stop_words='english')
#
# *   vectorizer_text.py also plays a role ***
# this is where the words are removed 
# remove_words = ["sara", "shackleton", "chris", "germani"]
# then vectorizer_text.py is re-run

for index, thisFloat in numpy.ndenumerate(myDecisionTreeClassifier.feature_importances_):
    # print("{} - {}".format(index, x))
    if thisFloat > 0.2:
        myHighCounter += 1
print("myHighCounter - {}".format(myHighCounter))

# print("mySaveindex - {}".format(mySaveindex))
#      mySaveindex[0] - 33614 - Right Answer
# print("type(mySaveindex) - {}\n".format(type(mySaveindex)))
#        type(mySaveindex) - <class 'tuple'>

print('\nEnd find_signature.py Python module\n')



