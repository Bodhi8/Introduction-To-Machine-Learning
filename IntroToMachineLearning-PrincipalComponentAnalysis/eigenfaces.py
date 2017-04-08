'''
Created on Apr 6, 2017

@author: Menfi
'''

"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  .. _LFW: http://vis-www.cs.umass.edu/lfw/

  original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html

"""

# print __doc__
# print ('__doc__')
print (__doc__)

from time import time
import logging
import pylab as pl
import numpy as np

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split


from sklearn.datasets import fetch_lfw_people

# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import RandomizedPCA
# from sklearn.decomposition import PCA

from sklearn.svm import SVC

print('\n.........................................Begin eigenfaces.py Python module\n')

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
# 2017-04-07 00:32:47,348 Loading LFW people faces from /Users/Menfi/scikit_learn_data/lfw_home
print()

###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

# for machine learning we use the data directly (as relative pixel
# position info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# print "Total dataset size:"
# print "n_samples: %d" % n_samples
# print "n_features: %d" % n_features
# print "n_classes: %d" % n_classes

print("Total dataset size: ")
print("n_samples - {}".format(n_samples))
print("n_features - {}".format(n_features))
print("n_classes - {}".format(n_classes))

###############################################################################
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

###############################################################################

# Principal Component Analysis -PCA actually happens here
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
# investigate component - f1 score relationship
# n_components = 150
n_components = 75

# print "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
print("n_components - {}".format(n_components))
print("X_train.shape[0] - {}".format(X_train.shape[0]))
print("Extracting the top {} eigenfaces from {} faces\n".format(n_components, X_train.shape[0]))

print('.............................................line 101')

t0 = time()

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
# pca = PCA(svd_solver = 'randomized', whiten = True).fit(X_train)
print("pca - {}".format(pca))
# pca - RandomizedPCA(copy=True, iterated_power=2, n_components=150, random_state=None, whiten=True)
# pca - PCA(copy=True, iterated_power='auto', n_components=None, random_state=None, svd_solver='randomized', tol=0.0, whiten=True)
print("type(pca) - {}\n".format(type(pca)))
# type(pca) - <class 'sklearn.decomposition.pca.RandomizedPCA'>
#        type(pca) - <class 'sklearn.decomposition.pca.PCA'>

# print "done in %0.3fs" % (time() - t0)
print("done in  - {}\n".format(time() - t0))


eigenfaces = pca.components_.reshape((n_components, h, w))

# print "Projecting the input data on the eigenfaces orthonormal basis"
print ("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# print "done in %0.3fs" % (time() - t0)
print("done in  - {}\n".format(time() - t0))

print('pca.components_[0]')
print(pca.components_[0])
print('pca.components_[1]')
print(pca.components_[1]) 
print()
print('pca.explained_variance_ratio_')
print(pca.explained_variance_ratio_)
print('pca.explained_variance_ratio_[0]')
print(pca.explained_variance_ratio_[0])
print('pca.explained_variance_ratio_[1]')
print(pca.explained_variance_ratio_[1])
print()


###############################################################################
# Train a SVM classification model

# print "Fitting the classifier to the training set"
print ("Fitting the classifier to the training set") 
t0 = time()
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
# print "done in %0.3fs" % (time() - t0)
print("done in  - {}\n".format(time() - t0))

# print "Best estimator found by grid search:"
print ("Best estimator found by grid search:")
print ('clf.best_estimator_')
print (clf.best_estimator_)
print()

###############################################################################
# Quantitative evaluation of the model quality on the test set

# print "Predicting the people names on the testing set"
print ("Predicting the people names on the testing set")
t0 = time()
y_pred = clf.predict(X_test_pca)
# print "done in %0.3fs" % (time() - t0)
# print ("done in {}\n" % (time() - t0))
print("done in  - {}\n".format(time() - t0))

print ("classification_report(y_test, y_pred, target_names=target_names)")
print (classification_report(y_test, y_pred, target_names=target_names))
print()

print ('confusion_matrix(y_test, y_pred, labels=range(n_classes))')
print (confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print()


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    
    print('\nBegin eigenfaces.py Module - plot_gallery function\n')

    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())
        
    print('\nEnd eigenfaces.py Module - plot_gallery function\n')



# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    print('\nBegin eigenfaces.py Module - title function\n')

    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    
    print('\nEnd eigenfaces.py Module - title function\n')
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

pl.show()
