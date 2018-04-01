import tensorflow as tf
import pandas as pd
import quandl
import math
import time
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import pickle
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
df = quandl.get("WIKI/GOOGL",authtoken="eQcjDb4dsnZXfyL1YzbF")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] *100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forcast_col = 'Adj. Close'
df.fillna(-9999,inplace= True)
forcast_out = int(math.ceil(0.01*len(df)))
df['Label'] = df[forcast_col].shift(-forcast_out)


X = np.array(df.drop(['Label'],1))
X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out]
df.dropna(inplace =True)
y = np.array(df['Label'])


X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
clf = LinearRegression(n_jobs=-1)
# #clf = svm.SVR()
# #clf = svm.SVR(kernel='poly')
clf.fit(X_train,Y_train)
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
accuray = clf.score(X_test,Y_test)
forcast_set = clf.predict(X_lately)
print(forcast_set, accuray,forcast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()