# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:40:41 2018

@author: NANCUH
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_feature = [5,6]
#for i in dataset:
#    for ii in dataset[i]:
#        plt.scatter(ii[0],ii[1],s=100,color=i)
#plt.scatter(new_feature[0],new_feature[1],s=100,color='b')
#plt.show()
def k_nearest_neighbors(data,predict,k=3):
    if len(data)>=k or k%2 ==0 :
        warnings.warn('K is set less change greater than dataset types or k is not to be even!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted((distances))[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result
result = k_nearest_neighbors(dataset,new_feature,k=3)
print(result)
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i)
plt.scatter(new_feature[0],new_feature[1],s=100,color=result)
plt.show()