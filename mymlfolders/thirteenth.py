# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:51:12 2018

@author: NANCUH
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing,cross_validation
import pandas as pd

df = pd.read_excel('titanic.xls')
orginal_df = pd.DataFrame(df)
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

df.drop(['ticket','sex','boat','home.dest'],1,inplace=True)
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)
labels=clf.labels_
cluster_centers = clf.cluster_centers_

orginal_df['cluster_group'] = np.nan

for i in range(len(X)):
    orginal_df['cluster_group'].iloc[i] = labels[i]
n_clusters = len(np.unique(labels))
survival_rates = {}

for i in range(n_clusters):
    temp_df = orginal_df[(orginal_df['cluster_group']==float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = float(len(survival_cluster))/float(len(temp_df))
    survival_rates[i] = survival_rate

print(survival_rates)
