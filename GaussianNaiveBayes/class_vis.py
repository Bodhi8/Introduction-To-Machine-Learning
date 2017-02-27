'''
Created on Feb 24, 2017

@author: Menfi
'''
 
 
import pylab
# from udacityplots import *
import warnings
from matplotlib.backends import pylab_setup
warnings.filterwarnings("ignore")

import matplotlib 
matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

#import numpy as np
#import matplotlib.pyplot as plt
#plt.ioff()

def prettyPicture(clf, X_test, y_test):
    
    print("\tBegin class_vis - prettyPicture function")

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

    # plt.savefig("test.png") # udacity location 
    # plt.savefig("/Users/Menfi/Documents/workspace/zzzzz/src/test.png")
    # plt.show("/Users/Menfi/Documents/workspace/zzzzz/src/test.png")
    pylab.show()
 
    print("\tEnd class_vis - prettyPicture function")

    
import base64
import json
import subprocess

def output_image(name, format, bytes):
    print("\tBegin class_vis - output_image function")

    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    # print (image_start+json.dumps(data)+image_end)
    # print file(r"c:\python\random_img\1.png", "rb").read()
    # print ((r'/Users/Menfi/Documents/workspace/zzzzz/src/test.png', 'rb').read()) 

    print("\tEnd class_vis - output_image function")
    
print("Begin class_vis")
print("End class_vis Module\n")
