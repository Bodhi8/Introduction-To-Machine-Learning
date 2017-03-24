#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster.k_means_ import KMeans
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.preprocessing.data import MinMaxScaler

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

largestESO = 0
smallestESO = 10000
largestSalary = 0
smallestSalary = 10000

for person, personValues in data_dict.items():
    print("person - {}".format(person))
    # print("type(person) - {}".format(type(person)))
    #        type(person) - <class 'str'>

    # print("personValues - {}".format(personValues))
    # print("type(personValues) - {}\n".format(type(personValues)))
    #        type(personValues) - <class 'dict'>

    for valueName, value in personValues.items():
        # print("valueName - {}".format(valueName))
        # print("type(valueName) - {}".format(type(valueName)))
        #        type(valueName) - <class 'str'>

        # print("value - {}".format(value))
        # print("type(value) - {}".format(type(value)))
        #        type - varies 
        if valueName == 'salary' or valueName == 'exercised_stock_options' or valueName == 'total_payments':
            print('{} - {}'.format(valueName, value))
        if valueName == 'exercised_stock_options' and value != 'NaN':
            if value > largestESO:
                largestESO = value
            if value < smallestESO:
                smallestESO = value
        
        if valueName == 'salary' and value != 'NaN':
            if value > largestSalary:
                largestSalary = value
            if value < smallestSalary:
                smallestSalary = value
                
        if valueName == 'exercised_stock_options' and value != 'NaN' and value > 990000 and value < 1100000 :
            print('................................xxxx{} - {}'.format(valueName, value))
        if valueName == 'salary' and value != 'NaN' and value > 190000 and value < 210000 :
            print('................................xxxx{} - {}'.format(valueName, value))

            
    print()
    
print("smallest exercised_stock_options - {}".format(smallestESO))
print("largest exercised_stock_options - {}".format(largestESO))
print("smallestSalary - {}".format(smallestSalary))
print("largestSalary - {}\n".format(largestSalary))

scaler = MinMaxScaler()
original_salary = numpy.array([[smallestSalary], [200000.], [largestSalary]])
rescaled_salary = scaler.fit_transform(original_salary)
print('rescaled_salary ->')
print(rescaled_salary)

original_exercised_stock_options = numpy.array([[smallestESO], [1000000.], [largestESO]])
rescaled_exercised_stock_options = scaler.fit_transform(original_exercised_stock_options)
print('rescaled_exercised_stock_options ->')
print(rescaled_exercised_stock_options)
 
# class video - Introduction To Machine Learning - Clustering - Quiz: Clustering Features
# What features will your clustering algorithm use?
# answer - 1.) salary, 2.) exercised_stock_options
### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments" # Clustering with 3 Features
poi  = "poi"
# features_list = [poi, feature_1, feature_2]
features_list = [poi, feature_1, feature_2, feature_3] # Clustering with 3 Features

# What does 'data' look like? 
# person - ELLIOTT STEVEN
# exercised_stock_options - 4890344
# salary - 170941
# total_payments - 211725 # Clustering with 3 Features

# data
#  poi            - salary         - exercised_stock_options
# [  0.00000000e+00   1.70941000e+05   4.89034400e+06]

data = featureFormat(data_dict, features_list )
print('\ndata ->')
print(data)
print()
poi, finance_features = targetFeatureSplit( data )

print('finance_features ->')
print(finance_features)
print(finance_features[0])
#    salary - exercised_stock_options
# [  428780.  1835558.]
print(finance_features[1])
print()

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
# for f1, f2 in finance_features: 
#   plt.scatter( f1, f2 )
# plt.show()

for f1, f2, f3 in finance_features: # Clustering with 3 Features
    plt.scatter( f1, f2 )
plt.show()


### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

myClassifier = KMeans(n_clusters=2)
pred = myClassifier.fit_predict(finance_features)
print("type(pred) - {}\n".format(type(pred)))
print("pred - {}\n".format(pred))


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    # print "no predictions object named pred found, no clusters to plot"
    print("no predictions object named pred found, no clusters to plot")
