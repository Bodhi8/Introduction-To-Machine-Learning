#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    myNewTupleList = []
    difference = 0
    
    print('\nBegin outlier_cleaner.py Python module\n')

    ### your code goes here
    
    # print("\ttype(predictions) - {}".format(type(predictions)))
    #         type(predictions) - <class 'numpy.ndarray'>
    # print("\ttype(ages) - {}".format(type(ages)))
    #          type(ages) - <class 'numpy.ndarray'>
    # print("\ttype(net_worths) - {}".format(type(net_worths)))
    # type(net_worths) - <class 'numpy.ndarray'>

    # predictions.shape - (90, 1), [[ 314.65206822], [ 314.65206822]]
    # ages_train.shape - (90, 1), [[57], [57]]
    # net_worths_train.shape - (90, 1), [[ 338.08951849], [ 344.21586776]]
    
    myZippedUpInput = zip(predictions, ages, net_worths)
    # print("\ttype(myZippedUpInput) - {}\n".format(type(myZippedUpInput)))
    #           type(myZippedUpInput) - <class 'zip'>
    
    for ii in myZippedUpInput:
        # print("\t\tii - {}".format(ii))
        # print("\t\ttype(ii) - {}".format(type(ii)))
        #            type(ii) - <class 'tuple'>
        # print("\t\tlen(ii) - {}".format(len(ii)))
        #            len(ii) - 3

        myPrediction, myAge, myNetWorths = ii
        # print("\t\tmyPrediction - {}".format(myPrediction))
        #            myPrediction - [ 314.65206822]
        # print("\t\ttype(myPrediction) - {}".format(type(myPrediction)))
        #            type(myPrediction) - <class 'numpy.ndarray'>

        # print("\t\tmyPrediction[0] - {}".format(myPrediction[0]))
        # print("\t\ttype(myPrediction[0]) - {}\n".format(type(myPrediction[0])))
        #            type(myPrediction[0]) - <class 'numpy.float64'>
        # print("\t\tmyAge[0] - {}".format(myAge[0]))
        # print("\t\tmyNetWorths[0] - {}".format(myNetWorths[0]))
        
        if myPrediction[0] > myNetWorths[0]:
            difference = myPrediction[0] - myNetWorths[0]
        else:
            difference = myNetWorths[0] - myPrediction[0]
        # print("\t\tdifference - {}".format(difference))

        myTempList = list(ii)
        myTempList.append(difference)
        myNewTuple = tuple(myTempList)
        
        # print("\t\tmyNewTuple - {}".format(myNewTuple))
        
        myNewTupleList.append(myNewTuple)
                         
        # print()
        # print()

    print(myNewTupleList)
    print("type(myNewTupleList) - {}\n".format(type(myNewTupleList)))
    
    sorted_by_fourth = sorted(myNewTupleList, key=lambda myNewTuple: myNewTuple[3])
    print('sorted_by_fourth ->')
    print(sorted_by_fourth)
    print()
#   print("\ntype(sorted_by_fourth) - {}".format(type(sorted_by_fourth)))
#            type(sorted_by_fourth) - <class 'list'>

    print("type(sorted_by_fourth[0]) - {}".format(type(sorted_by_fourth[0])))
    print("type(sorted_by_fourth[0][3]) - {}".format(type(sorted_by_fourth[0][3])))
    print("len(sorted_by_fourth) - {}\n".format(len(sorted_by_fourth)))
    
    print(sorted_by_fourth[0][3])
    print(sorted_by_fourth[1][3])
    print(sorted_by_fourth[2][3])
    print('...')
    print(sorted_by_fourth[87][3])
    print(sorted_by_fourth[88][3])
    print(sorted_by_fourth[89][3])
    print()
    
    myShortenedList =  sorted_by_fourth[0:81]
    print("myShortenedList[0] - {}".format(myShortenedList[0]))
    print("myShortenedList[80] - {}".format(myShortenedList[80]))
    print("len(myShortenedList) - {}".format(len(myShortenedList)))
    print("type(myShortenedList) - {}\n".format(type(myShortenedList)))

    print(myShortenedList[0][3])
    print(myShortenedList[1][3])
    print('...')
    print(myShortenedList[79][3])
    print(myShortenedList[80][3])
        
    print()
    
    for ii in myShortenedList:
#       print("type(ii) - {}\n".format(type(ii)))
#              type(ii) - <class 'tuple'>

        tempList = list(ii)

#       print("tempList - {}".format(tempList))
#              tempList - [array([ 294.34034565]), array([53]), array([ 368.29556369]), 73.95521804220931]
#       print("type(tempList) - {}\n".format(type(tempList)))
#              type(tempList) - <class 'list'>
        tempList.pop(0)
        # print("tempList - {}".format(tempList))
#              tempList - [array([53]), array([ 368.29556369]), 73.95521804220931]

        myNewTuple = tuple(tempList)
        # print("myNewTuple - {}\n".format(myNewTuple))
#       print("type(myNewTuple) - {}\n".format(type(myNewTuple)))
#              type(myNewTuple) - <class 'tuple'>

        cleaned_data.append(myNewTuple)


    print('End outlier_cleaner.py Python module\n')
    
    return cleaned_data

