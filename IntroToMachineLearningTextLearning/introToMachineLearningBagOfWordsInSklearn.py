
'''
Created on Mar 25, 2017

@author: Menfi
'''

import sys

# NLTK - National Language Tool Kit
# import nltk

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# import nltk
# import nltk.stem
# import nltk.stem.snowball

string1 = "hi Katie the self driving car will be late Best sebastian"
string2 = "Hi Sebastian the machine learning class will be great great great Best Katie"
string3 = "Hi Katie the machine learning class will be the most excellent" 

email_list = [string1, string2, string3]

vectorizer = CountVectorizer()

# print("vectorizer - {}\n".format(vectorizer))
# vectorizer - CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=1.0, max_features=None, min_df=1,
#         ngram_range=(1, 1), preprocessor=None, stop_words=None,
#         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#         tokenizer=None, vocabulary=None)

# print("type(vectorizer) - {}\n".format(type(vectorizer)))
#        type(vectorizer) - <class 'sklearn.feature_extraction.text.CountVectorizer'>

bag_of_words = vectorizer.fit(email_list)

# print("bag_of_words - {}\n".format(bag_of_words))
# bag_of_words - CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=1.0, max_features=None, min_df=1,
#         ngram_range=(1, 1), preprocessor=None, stop_words=None,
#         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#         tokenizer=None, vocabulary=None)

# print("type(bag_of_words) - {}\n".format(type(bag_of_words)))
#        type(bag_of_words) - <class 'sklearn.feature_extraction.text.CountVectorizer'>

# done AFTER stemming - stemming done before bag_of_words
# builds representation used in machine learning algorithm 
bag_of_words = vectorizer.transform(email_list)
print('bag_of_words - ')
print(bag_of_words)
print()

# - representation used in machine learning algorithm -
# tuple - integer 
# (1, 7)    1 -> document 1 - word 7 occurs 1 time 
# (1, 3)    1
# (1, 6)    3


# print("type(bag_of_words) - {}\n".format(type(bag_of_words)))
#        type(bag_of_words) - <class 'scipy.sparse.csr.csr_matrix'>

print("vectorizer.vocabulary_.get('great') - {}".format(vectorizer.vocabulary_.get('great')))
#        vectorizer.vocabulary_.get('great') - 6 - feature number

#print("type(vectorizer.vocabulary_.get('great')) - {}\n".format(type(vectorizer.vocabulary_.get('great'))))
#        type(vectorizer.vocabulary_.get('great')) - <class 'numpy.int64'>

# nltk.download()
myStopWords = stopwords.words('english')

print("myStopWords - {}".format(myStopWords))
# print("type(myStopWords) - {}\n".format(type(myStopWords)))
#        type(myStopWords) - <class 'list'>
print("len(myStopWords) - {}\n".format(len(myStopWords)))

# done BEFORE bag_of_words - stemming done first
myStemmer = SnowballStemmer('english')
# print("myStemmer - {}".format(myStemmer))
#        myStemmer - <nltk.stem.snowball.SnowballStemmer object at 0x1067b9a58>
# print("type(myStemmer) - {}\n".format(type(myStemmer)))
#        type(myStemmer) - <class 'nltk.stem.snowball.SnowballStemmer'>

print("myStemmer.stem('responsiveness') - {}".format(myStemmer.stem('responsiveness')))
print("myStemmer.stem('responsivity')\t - {}".format(myStemmer.stem('responsivity')))
print("myStemmer.stem('unresponsive')\t - {}\n".format(myStemmer.stem('unresponsive')))
#      myStemmer.stem('responsiveness') - respons
#  print("type(myStemmer.stem('responsiveness')) - {}\n".format(type(myStemmer.stem('responsiveness'))))
#        type(myStemmer.stem('responsiveness')) - <class 'str'>

for path in sys.path:
    print("path - {}".format(path))
    
# path - /Users/Menfi/Documents/workspace/IntroToMachineLearning-BagOfWordsInSklearn/src
# path - /Users/Menfi/Documents/workspace/IntroToMachineLearning-BagOfWordsInSklearn/src
# path - /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5
# path - /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/plat-darwin
# path - /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/lib-dynload
# path - /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages
# path - /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/xlrd-1.0.0-py3.5.egg
# path - /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/requests
# path - /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/requests/packages
# path - /Library/Frameworks/Python.framework/Versions/3.5/lib/python35.zip








