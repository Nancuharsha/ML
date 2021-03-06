# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 17:52:16 2018

@author: NANCUH
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing
import pandas as pd

#X =np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[1,3],[8,9],[0,3],[5,4],[6,4]])
#plt.scatter(X[:,0],X[:,1],s=150,linewidths=5)
#plt.show()

colors = 10*["g","r","c","b","k"]

class K_Means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k=2
        self.tol=tol
        self.max_iter=max_iter
        
    def fit(self,data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
        for i in range(self.max_iter):
            self.classifications ={}
            for i in range(self.k):
                self.classifications[i] = []
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            pre_centroids =dict(self.centroids)
            
            for classification in self.classifications:
                #pass
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            
            optimzed = True
            for c in self.centroids:
                org_cen = pre_centroids[c]
                cur_cen = self.centroids[c]
                if np.sum((cur_cen-org_cen)/org_cen * 100.0) > self.tol:
                    optimzed=False
            if optimzed:
                break
            
        
        
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
#clf = K_Means(k=2)
#clf.fit(X)
#
#for centroid in clf.centroids:
#    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker='o',color="k",s=150,linewidths=5)
#
#for classification in clf.classifications:
#    color = colors[classification]
#    for featureset in clf.classifications[classification]:
#        plt.scatter(featureset[0],featureset[1],marker="x",color = color,s=150,linewidths=5)

#unknowns = np.array([[1,3],[8,9],[0,3],[5,4],[6,4]])
#
#for unknown in unknowns:
#    classification = clf.predict(unknown)
#    plt.scatter(unknown[0],unknown[1],marker="*",color = colors[classification],s=150,linewidths=5)

#plt.show()















df = pd.read_excel('titanic.xls')
#print(df.head())
df.drop(['body','name'],1,inplace = True)
#print(df.head())
#df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            columns_contents = df[column].values.tolist()
            unique_elements =set(columns_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int,df[column]))
    return df
df = handle_non_numerical_data(df)
#print(df.head())

df.drop(['ticket','home.dest'],1,inplace=True)
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = K_Means()
clf.fit(X)

#print(clf.centroids)
correct = 0
for i in range(len(X)):
    predict_me= np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct+=1

print(float(correct)/float(len(X)))







