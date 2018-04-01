# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:53:55 2018

@author: NANCUH
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('fivethirtyeight')
#xs = np.array([1,2,3,4,5,6],dtype=np.float64)
#ys =np.array([5,4,6,5,6,7],dtype = np.float64)
#plt.scatter(xs,ys)
#plt.show()
def create_dataset(hm,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)


def best_fit_slope_and_intercept(xs,ys):
    m = ((mean(xs)*mean(ys))-mean(xs*ys))/(mean(xs)*mean(xs)-mean(xs**2))
    b = (mean(ys)-m*(mean(xs)))
    return m,b
def square_error(y_org,y_line):
    return sum((y_line-y_org)**2)
def coefficient_of_determination(y_org,y_line):
    y_mean = [mean(y_org) for x in y_org]
    square_error_regr = square_error(y_org,y_line)
    square_error_y_mean = square_error(y_org,y_mean)
    return 1- (square_error_regr/square_error_y_mean)

xs ,ys = create_dataset(40,80,2,correlation='pos')

m,b = best_fit_slope_and_intercept(xs,ys)
predict_x = 8
predict_y = (m*predict_x) +b
#for x in xs:
#    regression_line.append((m*x+b))
regression_line = [(m*x+b) for x in xs]
print(coefficient_of_determination(ys,regression_line))
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,s=100,color='g')
plt.plot(xs,regression_line)
plt.show()