from ast import increment_lineno
from xmlrpc.client import DateTime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


"""df = pd.read_csv("TRAFFIC123.csv")

data = df.drop(['DateTime','A','B'], axis=1)
data.to_csv('TrafficFinal.csv')"""


df = pd.read_csv("TrafficFinal.csv")
x = df['time']
x= x.values.reshape(-1, 1)
y = df['Vehicles']
y = y.values.reshape(-1,1)


clf = LinearRegression()
clf.fit(x,y)

inarray = np.array([[19]]) 

print('No of vehicles :',int(clf.predict(inarray)))




