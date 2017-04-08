'''
Created on Apr 6, 2017

@author: Menfi
'''

from sklearn.decomposition import PCA

def doPCA():
# def doPCA(data):
    print('\tBegin doPCA() function\n')

    pca = PCA(n_components = 2)
    print("pca - ")
    print(pca)
    # PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
    print()

    # print("\ttype(pca) - {}\n".format(type(pca)))
    #        type(pca) - <class 'sklearn.decomposition.pca.PCA'>


    # pca.fit(data)
    print('\tEnd doPCA() function\n')

    return pca



print('\nBegin introMachineLearningPrincipalComponentAnalysis.py Python module\n')

pca = doPCA()
# pca = doPCA(data)
# print("pca - ")
# print(pca)
# PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)

print('pca.explained_variance_ratio_')
# print(pca.explained_variance_ratio_)
print()

# access first and second Principal Components 
# first_pc = pca.components_[0]
# second_pc = pca.components_[1]

print('End introMachineLearningPrincipalComponentAnalysis.py Python module\n')


