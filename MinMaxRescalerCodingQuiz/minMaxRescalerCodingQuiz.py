'''
Created on Mar 24, 2017

@author: Menfi
'''

""" quiz materials for feature scaling clustering """

print()
myRescaledList = []

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    # print("type(arr) - {}\n".format(type(arr)))
    #        type(arr) - <class 'list'>
    
    # print("arr[0] - {}".format(arr[0]))
    myMinNumber = min(arr)
    print("myMinNumber - {}".format(myMinNumber))
    myMaxNumber = max(arr)


    
    for inputNumber in arr:
        myRescaledNumber = (inputNumber - myMinNumber) / (myMaxNumber - myMinNumber)
        myRescaledList.append(myRescaledNumber)
    
    # print(myRescaledList)
    return myRescaledList
    # return None

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
# print featureScaling(data)
print("featureScaling(data) - {}\n".format(featureScaling(data)))


