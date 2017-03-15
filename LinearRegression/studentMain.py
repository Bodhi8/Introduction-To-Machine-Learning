'''
Created on Mar 14, 2017

@author: Menfi
'''


import matplotlib.pyplot as plt
from studentRegression import studentReg
from class_vis import prettyPicture, output_image

from ages_net_worths import ageNetWorthData

print('\nBegin studentMain.py - studentReg function\n')

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

print("\nages_train[0:2,0:3] - {}".format(ages_train[0:2,0:3]))
print("\nages_train[0:2] - ")
print(ages_train[0:2,])

print("\ntype(ages_train) - {}".format(type(ages_train)))
#        type(ages_train) - <class 'numpy.ndarray'>


print("\nnet_worths_train[0:2,0:3] - {}".format(net_worths_train[0:2,0:3]))
print("\nnet_worths_train[0:2] - ")
print(net_worths_train[0:2,])

print("type(net_worths_train) - {}\n".format(type(net_worths_train)))
#      type(net_worths_train) - <class 'numpy.ndarray'>



# Call the studentRegression.py function studentReg passing ages_train, net_worths_train
reg = studentReg(ages_train, net_worths_train)


plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2) # did not seem to make a difference 
plt.xlabel("ages")
plt.ylabel("net worths")
plt.show()


plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())

print('End studentMain.py - studentReg function\n')
