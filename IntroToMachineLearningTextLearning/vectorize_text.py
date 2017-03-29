#!/usr/bin/python

import os
import pickle
import re
# import sys

# sys.path.append( "../tools/" )
# from parse_out_email_text import parseOutText

import parse_out_email_text
from parse_out_email_text import parseOutText

myTempString = ''

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
# GET TYPE OF FILE - unix command line
# $ file from_sara.txt - ASCII file 
# from_sara.txt: ASCII text

# GET CONTENTS OF FILE - unix command line
# $ head from_sara.txt - A list of pathnames to the actual emails
#        maildir/bailey-s/deleted_items/101.
#        maildir/bailey-s/deleted_items/106.

# print("from_sara - {}".format(from_sara))
#        from_sara - <_io.TextIOWrapper name='from_sara.txt' mode='r' encoding='US-ASCII'>
# print("type(from_sara) - {}\n".format(type(from_sara)))
#        type(from_sara) - <class '_io.TextIOWrapper'>

from_chris = open("from_chris.txt", "r")
# GET TYPE OF FILE - unix command line
# $ file from_chris.txt  - ASCII file 
# from_chris.txt: ASCII text

# GET CONTENTS OF FILE - unix command line
# $ head -2 from_chris.txt  - A list of pathnames to the actual emails
# maildir/donohoe-t/inbox/253.
# maildir/germany-c/_sent_mail/1.

# print("from_chris - {}".format(from_chris))
#        from_chris - <_io.TextIOWrapper name='from_chris.txt' mode='r' encoding='US-ASCII'>
# print("type(from_chris) - {}\n".format(type(from_chris)))
#        type(from_chris) - <class '_io.TextIOWrapper'>

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

for ii in from_sara:
    pass
    # print(ii)
    # maildir/bailey-s/deleted_items/101. - carriage return newline still needs to be stripped out see below
    # maildir/bailey-s/deleted_items/106. - carriage return newline still needs to be stripped out see below
    
from_sara.seek(0) # interated through - TextIOWrapper - back to beginning of file 

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
# for name, from_person in [("sara", from_sara)]:
    print("\nname - {}".format(name))
    #      name - sara
    # print("type(name) - {}".format(type(name)))
    #      type(name) - <class 'str'>
    
    print("from_person - {}\n".format(from_person))
    # from_person - <_io.TextIOWrapper name='from_sara.txt' mode='r' encoding='US-ASCII'>
    # print("type(from_person) - {}".format(type(from_person)))
    #        type(from_person) - <class '_io.TextIOWrapper'>

    temp_counter = 0 # FIX?

    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        # if temp_counter < 200: FIX
        if temp_counter < 1000000: # FIX
        # if temp_counter < 30: # FIX
            
            # print("path -")
            # print(path)
            # print(len(path))
            # print("type(path) - {}".format(type(path)))
            #        type(path) - <class 'str'>

            # print('path[:-1]')
            # print(path[:-1])
            # print(len(path[:-1]))
            # print("type(path[:-1]) - {}\n".format(type(path[:-1])))
            #      type(path[:-1]) - <class 'str'>

            # actual work HERE
            # path string edit - 1 of 2 - prepend  .. 
            # path string edit - 2 of 2 - remove newline, carriage return  
            path = os.path.join('..', path[:-1])
            # print("path - ")
            # print(path)
            # print()

            # Get the email contents, using the path to the email contents (working with one email at a time) 
            email = open(path, "r")
            
            # print("email")
            # print(email)
            # <_io.TextIOWrapper name='../maildir/germany-c/_sent_mail/10.' mode='r' encoding='US-ASCII'>
            # print(type(email))
            # <class '_io.TextIOWrapper'> 
                 
            ### use parseOutText to extract the text from the opened email
            text = parseOutText(email)
            # print("text returned - {}\n".format(text))
            # print("type(text) - {}\n".format(type(text)))
            #        type(text) - <class 'str'>

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            remove_words = ["sara", "shackleton", "chris", "germani"]
            # testString ='firstS sara last shackleton firstC chris country germani' 
            
            # do not use stopwords variable name - conflict with out of box variable nltk.corpus
            stop_words_local = ["sara", "shackleton", "chris", "germani"]

            fixedString = ''
            
            # thank chris 7138534743 - thank 7138534743
            #                             - 1 space left behind
            for testWord in text.split():
            # for testWord in testString.split(): # testing only
                if testWord in stop_words_local:
                    pass
                    # print("stop_words_local - {}".format(stop_words_local))
                else:
                    # print("testWord -XX{}XX".format(testWord))
                    fixedString = fixedString + testWord + ' '
                    # fixedString = fixedString + testWord
                    
            # print("fixedString - {}".format(fixedString))
            
            # thank chris 7138534743
            # thank  7138534743 - TWO SPACES left behind .replace leaves two spaces behind 
            #      12 
            # instructions said do it this way
            for e in remove_words:
                text =  text.replace(e,"")
            # print("text - {}\n".format(text))

            ### append the text to word_data
            # append the 'fixed' text of this email 
            # print("type(word_data) - {}".format(type(word_data)))
            #        type(word_data) - <class 'list'>
            # word_data.append(fixedString)
            word_data.append(text)
            # print("word_data - {}".format(word_data))
            print("len(word_data) - {}\n".format(len(word_data)))
            
            if len(word_data) > 200:
                # temp_counter = 1000000
                print(word_data[152])

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris            
            # print("type(from_data) - {}".format(type(from_data)))
            #        type(from_data) - <class 'list'>
            
            # print("name - {}\n".format(name))
            if name == 'sara':
                from_data.append(0)
            if name == 'chris':
                from_data.append(1)
                
            #print("from_data - {}\n".format(from_data))
            email.close()

            
# stemming completed in parse_out_email_text.py
# from nltk.stem.snowball import SnowballStemmer
# this code for example only not needed here 
# 
# # done BEFORE bag_of_words - stemming done first
# myStemmer = SnowballStemmer('english')
# print("myStemmer - {}\n".format(myStemmer))
# print("type(myStemmer) - {}\n".format(type(myStemmer)))
# 
for myString in word_data:
    # print("myString - {}".format(myString))
    # print("type(myString) - {}".format(type(myString)))
    #        type(myString) - <class 'str'>
    myStemmedWordList = myString.split()
      # print("myListOfWords - {}".format(myListOfWords))
      # print("type(myListOfWords) - {}".format(type(myListOfWords)))
      #        type(myListOfWords) - <class 'list'>
    for myStemmedWord in myStemmedWordList:
        # print("myStemmedWord - {}".format(myStemmedWord))  
        # print("type(myWord) - {}".format(type(myWord)))
        pass
    # print()
    

from nltk.corpus import stopwords    # instructions said do not use 
# myStopWords = stopwords.words('english') # nltk stop words do not use !!
# print("myStopWords - {}".format(myStopWords))
# print("type(myStopWords) - {}\n".format(type(myStopWords)))
#        type(myStopWords) - <class 'list'>

# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer # instructions said use this vectorizer 
# vectorizer = CountVectorizer(stop_words = myStopWords, lowercase = True)
vectorizer = TfidfVectorizer(stop_words = 'english', lowercase = True) # sklearn stop_words - do it this way!! 

# print("vectorizer - {}".format(vectorizer))
# vectorizer - CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=1.0, max_features=None, min_df=1,
#         ngram_range=(1, 1), preprocessor=None,
#         stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', '...aven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'],
#         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#         tokenizer=None, vocabulary=None) 
# print("type(vectorizer) - {}\n".format(type(vectorizer)))
#        type(vectorizer) - <class 'sklearn.feature_extraction.text.CountVectorizer'>

# bag_of_words = vectorizer.fit(word_data)
# print("bag_of_words - {}".format(bag_of_words))
# bag_of_words - CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=1.0, max_features=None, min_df=1,
#         ngram_range=(1, 1), preprocessor=None,
#         stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', '...aven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'],
#         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#         tokenizer=None, vocabulary=None)
# print("type(bag_of_words) - {}\n".format(type(bag_of_words)))
#        type(bag_of_words) - <class 'sklearn.feature_extraction.text.CountVectorizer'>

vectorizer.fit_transform(word_data) # fit transform convenience 
# print("transformed_bag_of_words - {}\n".format(transformed_bag_of_words))
#     (0, 91)     1
#     (0, 230)    1
#     (0, 242)    2
# print("type(transformed_bag_of_words) - {}\n".format(type(transformed_bag_of_words)))
#        type(transformed_bag_of_words) - <class 'scipy.sparse.csr.csr_matrix'>

# myShape = transformed_bag_of_words.get_shape()
# print("myShape - {}".format(myShape))
# print("type(myShape) - {}\n".format(type(myShape)))
#      type(my_feature_names) - <class 'list'>


# print("transformed_bag_of_words.nnz - {}\n".format(transformed_bag_of_words.nnz))
# print("type(transformed_bag_of_words.nnz) - {}\n".format(type(transformed_bag_of_words.nnz)))
#        type(transformed_bag_of_words.nnz) - <class 'int'>

my_feature_names = vectorizer.get_feature_names() # call the get_feature_names() function on the vectorizer!!  
# print("my_feature_names - {}".format(my_feature_names))
#        my_feature_names - ['000', '004', '012246', '0152', '0153', '01
# print("type(my_feature_names) - {}\n".format(type(my_feature_names)))
#      type(my_feature_names) - <class 'list'>
print("len(my_feature_names) - {}\n".format(len(my_feature_names)))

print('my_feature_names[34597]')
print(my_feature_names[34597])

print("emails processed")
# print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )


### in Part 4, do TfIdf vectorization here


