#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

print('\nBegin enron_outliers.py Python module\n')


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )

# print("data_dict - {}\n".format(data_dict))
# print("type(data_dict) - {}\n".format(type(data_dict)))
#         type(data_dict) - <class 'dict'>

for dictionaryKey, dictionaryValue in data_dict.items():
    # print k, 'corresponds to', v
    # print("dictionaryKey {} - corresponds to dictionaryValue - {}".format(dictionaryKey, dictionaryValue))
    # print("dictionaryKey - {}".format(dictionaryKey))
    # print("type(dictionaryKey) - {}\n".format(type(dictionaryKey)))
    #        type(dictionaryKey) - <class 'str'>

    # print("dictionaryValue - {}".format(dictionaryValue))
    # print("type(dictionaryValue) - {}\n".format(type(dictionaryValue)))
    #        type(dictionaryValue) - <class 'dict'>
    pass

for person, personValues in data_dict.items():
    # print k, 'corresponds to', v
    # print("dictionaryKey {} - corresponds to dictionaryValue - {}".format(dictionaryKey, dictionaryValue))
    print("person - {}".format(person))
    # print("type(person) - {}\n".format(type(person)))
    # type(person) - <class 'str'>

    # print("personValues - {}".format(personValues))
    # print("type(personValues) - {}\n".format(type(personValues)))
    #        type(personValues) - <class 'dict'>
    
    for valueName, value in personValues.items():
        # print("valueName - value pairing - {} - {}".format(valueName, value))
        if valueName == 'salary':
            print("{} - {}".format(valueName, value))
        if valueName == 'bonus':
            print("{} - {}".format(valueName, value))
            if value != 'NaN' and int(value) > 5000000:
                print("person greater than 5 million - {}".format(person))

    print()
'''   
data_dict 
    person - FASTOW ANDREW S
    salary - 440698
    bonus - 1300000
''' 
    
features = ["salary", "bonus"]

# cleaned up some both salary and bonus look right 
data_dict.pop('TOTAL')
data = featureFormat(data_dict, features)
print("type(data) - {}\n".format(type(data)))
#      type(data) - <class 'numpy.ndarray'>

print("data")
print(data)
print()
print("data[0:2,0:2]")
print(data[0:2,0:2]) 
print()

'''
data[0:2,0:2]
     salary    bonus
[[  265214.   600000.]
 [  267102.  1200000.]]
'''


#print("type(data) - {}\n".format(type(data)))
#      type(data) - <class 'numpy.ndarray'>

### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
    
print("type(salary) - {}".format(type(salary)))
print("type(bonus) - {}".format(type(bonus)))


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

print('\nEnd enron_outliers.py Python module\n')
