#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

# from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors

print('\nBegin your_algorithm.py Python module\n')

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

# this example of ** import matplotlib.pyplot as plt ** plt.show() is from udacity.com, not mine - works!
#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary



# !! sklearn.neighbors.KNeighborsClassifier

myKNeighborsClassifier = sklearn.neighbors.KNeighborsClassifier()
print("myKNeighborsClassifier - ")
print(myKNeighborsClassifier)
print("\ntype(myKNeighborsClassifier) - {}\n".format(type(myKNeighborsClassifier)))

myKNeighborsClassifier.fit(features_train,labels_train)

pred = myKNeighborsClassifier.predict(features_test)

myKNeighborsClassifier_score = myKNeighborsClassifier.score(features_test,labels_test)
print("myKNeighborsClassifier_score - {}".format(myKNeighborsClassifier_score))
# print("type(myKNeighborsClassifier_score) - {}\n".format(type(myKNeighborsClassifier_score)))

try:
    pass
    # prettyPicture(clf, features_test, labels_test)
    prettyPicture(myKNeighborsClassifier, features_test, labels_test)
except NameError:
    pass

print('\nEnd your_algorithm.py Python module\n')
