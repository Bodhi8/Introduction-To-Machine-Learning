'''
Created on Feb 24, 2017

@author: Menfi
'''

import random

print("\nBegin prep_terrain_data.py Module")

# def makeTerrainData(n_points = 1000):
def makeTerrainData(n_points = 1000):
    
    print("\tBegin prep_terrain_data.py makeTerrainData function")
    random.seed(42)
    # generate Pyhton list(s)
    
    grade = [random.random() for ii in range(0,n_points)]
    # [0.6394267984578837, 0.025010755222666936, ...
    print("\tgrade is {}".format(grade))

    bumpy = [random.random() for ii in range(0,n_points)]
    # [0.6766994874229113, 0.8921795677048454, ...
    print("\tbumpy is {}".format(bumpy))

    error = [random.random() for ii in range(0,n_points)]
    # [0.21863797480360336, 0.5053552881033624, ...
    print("\terror is {}\n".format(error))
   
    y = [round(  (grade[ii] * bumpy[ii])  + 0.3 +  (0.1 * error[ii])  ) for ii in range(0, n_points)]
    print("\ty is {}".format(y))
    # y is [1, 0, 0, 0, 1, 1, 1, 0, 1, 0]
    print("\tlen(y) is {}\n".format(len(y))) # len(y) is 10
    
    # modify y list based on grade and bumpy list
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0
            
    print("\ttransformed y is {}".format(y))
    print("\tlen transformed (y) is {}\n".format(len(y)))
    
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    print("\tX - [grade, bumpy pairs] - is {}".format(X))
    # X is [[0.6394267984578837, 0.27502931836911926], [0.025010755222666936, 0.22321073814882275]]
    print("\tlen(X) is {}\n".format(len(X)))
    # print("\ttype(X) is {}".format(type(X))) # list (actually list of lists ... print(X[0][1]) ... )
    
    split = int(0.75 * n_points)
    print("\tsplit is {}\n".format(split))
    # print("\ttype(split) is {}\n".format(type(split))) # int
    
    # X_train first 75% or 3/4 of X list of lists
    X_train = X[0:split]
    print("\tX_train is {}".format(X_train))
    # print("\ttype(X_train) is {}\n".format(type(X_train))) # list

    # X_test last 25% or 1/4 of X list of lists    
    X_test  = X[split:]
    print("\tX_test is {}\n".format(X_test))
    # print("\ttype(X_test) is {}".format(type(X_test))) # list
    
    # first 3/4 of y
    y_train = y[0:split]
    print("\ty_train is {}".format(y_train))
    # print("\ttype(X_test) is {}".format(type(X_test))) # list

    # last 1/4 of y
    y_test  = y[split:]
    print("\ty_test is {}".format(y_test))
    # print("\ttype(y_test) is {}".format(type(y_test))) # list


    # print("\ttype(y) is {}".format(type(y)))
    # print("\ty.__class__.__name__ is {}\n".format(y.__class__.__name__))
    
    print("\tEnd prep_terrain_data.py makeTerrainData function\n")

    return X_train, y_train, X_test, y_test


# makeTerrainData()

print("Endprep_terrain_data.py Module\n")
