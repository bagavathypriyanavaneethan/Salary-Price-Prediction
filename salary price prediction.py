# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:23:17 2020

@author: Bagavathi Priya
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

#Reading the dataset(csv file)
df=pd.read_csv(r'D:/Education/hiring.csv')
df.head()

#Filling the missing value
df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)

#Separating feature(x) and label(y)
x=df.iloc[:,:3]
y=df.iloc[:,-1]

#Function for converting text value in field to numerical value
def convert(text):
    dic={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,
         'zero':0,'ten':10,'eleven':11,0:0}
    return dic[text]

x['experience']=x['experience'].apply(lambda x:convert(x))


#Model creation
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x,y)


#Saving the model
pickle.dump(regressor,open('D:/Education/Salary price prediction/model.pkl','wb'))

#Load the model
model=pickle.load(open('model.pkl','rb'))

#Model prediction
print(model.predict([[3,5,9]]))






