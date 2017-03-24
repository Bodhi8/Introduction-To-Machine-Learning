#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
# import sys
# sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import sklearn.cluster
from sklearn.cross_validation import PredefinedSplit

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
# print("\ntype(data_dict) - {}\n".format(type(data_dict)))
#          type(data_dict) - <class 'dict'>

### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

max_saved_exercised_stock_options = 0
min_saved_exercised_stock_options = 10000
max_saved_salary = 0
min_saved_salary = 10000

for key, value in data_dict.items():
    print("key - {}".format(key))
    # print("value - {}\n".format(value))
    for valueKey, valueValue in value.items():
        # print(valueKey)
        if valueKey == 'salary' or valueKey == 'exercised_stock_options' or valueKey == 'total_payments' :
            # print("value - {}".format(value))
            # print("valueKey - {}".format(valueKey))
            # print("valueValue - {}\n".format(valueValue))
            # print("{} - {}".format(valueKey, valueValue))
            if valueKey == 'exercised_stock_options' and valueValue != 'NaN':
                print("{} - {}".format(valueKey, valueValue))
                # print("type(valueValue) - {}".format(type(valueValue)))
                if valueValue > max_saved_exercised_stock_options:
                    max_saved_exercised_stock_options = valueValue
                if min_saved_exercised_stock_options > valueValue:
                    min_saved_exercised_stock_options = valueValue
            if valueKey == 'salary' and valueValue != 'NaN':
                print("{} - {}".format(valueKey, valueValue))
                # print("type(valueValue) - {}".format(type(valueValue)))
                if valueValue > max_saved_salary:
                    max_saved_salary = valueValue
                if min_saved_salary > valueValue:
                    min_saved_salary = valueValue
    print()
                    
print("max_saved_exercised_stock_options - {}".format(max_saved_exercised_stock_options))
print("min_saved_exercised_stock_options - {}".format(min_saved_exercised_stock_options))
print("min_saved_salary - {}".format(min_saved_salary))
print("max_saved_salary - {}\n".format(max_saved_salary))

    
    
### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
# features_list = [poi, feature_1, feature_2, feature_3]

data = featureFormat(data_dict, features_list )
print('data')
print(data)
# print("type(data) - {}\n".format(type(data)))
#        type(data) - <class 'numpy.ndarray'>

# data examples
# key - LAY KENNETH L
# salary - 1072321
# exercised_stock_options - 34348384
# total_payments - 103559793

# key - BELDEN TIMOTHY N
# salary - 213999
# exercised_stock_options - 953136
# total_payments - 5501630

# What does 'data' look like
#   poi             -  salary        -  exercised_stock_options - total_payments 
# [  1.00000000e+00   1.07232100e+06   3.43483840e+07] - LAY KENNETH L
# [  1.00000000e+00   1.07232100e+06   3.43483840e+07   1.03559793e+08]

# [  1.00000000e+00   2.13999000e+05   9.53136000e+05] - BELDEN TIMOTHY N  
# [  1.00000000e+00   2.13999000e+05   9.53136000e+05   5.50163000e+06]

poi, finance_features = targetFeatureSplit( data )
# print("type(finance_features) - {}\n".format(type(finance_features)))
#        type(finance_features) - <class 'list'>

print("\npoi - {}\n".format(poi))
# poi - [0.0, 0.0, 0.0, 0.0, 0.0,
# print("type(poi) - {}\n".format(type(poi)))
#      type(poi) - <class 'list'>

# What does finance_features look like? 
print("finance_features - <class 'list'>")
print(finance_features)
# [array([ 184899.,       0.]),              array([  130724.,  2282768.]), 
# [array([       0.,  1753766.,        0.]), array([      0.,  759557.,       0.]), 
#         salary  - exercised_stock_options - total_payments
# array([  1072321.,         34348384.]), - LAY KENNETH L
# array([  1.07232100e+06,   3.43483840e+07,   1.03559793e+08]),

#         salary  - exercised_stock_options - total_payments
# array([ 213999.,  953136.]), - BELDEN TIMOTHY N
# array([  213999.,   953136.,  5501630.]

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
# for f1, f2, f3 in finance_features:
for f1, f2 in finance_features:
    pass

    # print("f1 (salary) - {}".format(f1))
    # print("type(f1) - {}".format(type(f1)))
    # type(f1) - <class 'numpy.float64'>

    # print("f2 (exercised_stock_options) - {}".format(f2))
    # print("type(f2) - {}\n".format(type(f2)))
    # type(f1) - <class 'numpy.float64'>

    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

myKMeans = sklearn.cluster.KMeans(n_clusters = 2)
# print("type(myKMeans) - {}\n".format(type(myKMeans)))
#            type(myKMeans) - <class 'sklearn.cluster.k_means_.KMeans'>

# print("type(finance_features) - {}\n".format(type(finance_features)))
#        type(finance_features) - <class 'list'>

myKMeans.fit(finance_features)
# pred = myKMeans.predict(finance_features)

pred = myKMeans.fit_predict(finance_features)

print('pred')
print(pred)
# print("type(pred) - {}\n".format(type(pred)))
# type(pred) - <class 'numpy.ndarray'>

# print('finance_features')
# print(finance_features) - no change
### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print ("\nno predictions object named pred found, no clusters to plot")
