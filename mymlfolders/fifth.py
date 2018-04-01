# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:13:43 2018

@author: NANCUH
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:40:41 2018

@author: NANCUH
"""

import numpy as np
from math import sqrt
import pandas as pd
import warnings
import random
from collections import Counter


#dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
#new_feature = [5,6]
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
    confidence = float(Counter(votes).most_common(1)[0][1])/float(k)
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result,confidence
#result = k_nearest_neighbors(dataset,new_feature,k=3)
#print(result)
#for i in dataset:
#    for ii in dataset[i]:
#        plt.scatter(ii[0],ii[1],s=100,color=i)
#plt.scatter(new_feature[0],new_feature[1],s=100,color=result)
#plt.show()
accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?',-99999,inplace = True)
    df.drop(['id'],1,inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)
    
    test_size = 0.4
    train_set = {2:[],4:[]}
    test_set = {2:[],4:[]}
    
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]
    
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    correct = 0
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote,confidence = k_nearest_neighbors(train_set,data,k=5)
            if group == vote:
                correct+=1
                #print(confidence)
            total+=1
    #print(correct,total)
    print('accuracy:',float(correct)/float(total))
    accuracies.append(float(correct)/float(total))
print(sum(accuracies)/len(accuracies))
    
    