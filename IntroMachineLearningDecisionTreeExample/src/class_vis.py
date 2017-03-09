'''
Created on Mar 8, 2017

@author: Menfi
'''

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import sklearn
import sklearn.svm
import sklearn.naive_bayes

def prettyPicture(clf, X_test, y_test):
    
    print("\tBegin class_vis - prettyPicture function\n")
    print("\ttype(clf) - {}".format(type(clf)))
    print("\tclf.__class__ - {}\n".format(clf.__class__))
    print("\tisinstance(clf, sklearn.naive_bayes.GaussianNB) - {}".format(isinstance(clf, sklearn.naive_bayes.GaussianNB)))
    print("\tisinstance(clf, sklearn.svm.classes.SVC) - {}\n".format(isinstance(clf, sklearn.svm.classes.SVC)))
    
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plt.pcolormesh(xx, yy, Z, cmap=pylab.cm.seismic)
    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic_r)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "g", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    
    # print("clf.__class__.__name - {}".format(clf.__name__))
    
    if isinstance(clf, sklearn.naive_bayes.GaussianNB):
        plt.title("Naive Bayes Gaussian - Terrain Classifier")
    elif isinstance(clf, sklearn.svm.classes.SVC):
        plt.title("SVM - Support Vector Machines Classifier - SVM - Terrain Classifier")

    # plt.savefig("test.png") # udacity location 
    # plt.savefig("/Users/Menfi/Documents/workspace/zzzzz/src/test.png")
    # plt.show("/Users/Menfi/Documents/workspace/zzzzz/src/test.png")
    plt.show() # works
    # pylab.show() # works
 
    print("\tEnd class_vis - prettyPicture function")
    
'''
def prettyPicture(clf, X_test, y_test):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")
    
'''

import base64
import json
import subprocess

def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    
    # print image_start+json.dumps(data)+image_end
    print(image_start+json.dumps(data)+image_end)

