'''
Created on Mar 14, 2017

@author: Menfi
'''

from sklearn import linear_model


def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    
    ### your code goes here!
    
    print('\nBegin studentRegression.py - studentReg function\n')

    
    print("ages_train - ")
    print(ages_train)
    print("type(ages_train) - {}\n".format(type(ages_train)))
    
    print("net_worths_train - ")
    print(net_worths_train)
    print("type(net_worths_train) - {}\n".format(type(net_worths_train)))
    
    reg = linear_model.LinearRegression()
    reg.fit (ages_train, net_worths_train)
    # reg.fit (net_worths_train, ages_train) # NO DID NOT WORK

    
    # reg = 1
    
    print('End studentRegression.py - studentReg function\n')

    
    
    return reg