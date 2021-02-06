# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:00:27 2021

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
LabelEncoder_X = LabelEncoder()
X[:,3]= LabelEncoder_X.fit_transform(X[:,3])

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))


#avoiding dummy variable trap
X = X[:, 1:]


#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state = 0)


#fit multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting test set results
y_pred = regressor.predict(X_test)

#building model using Backward Elimination
import statsmodels.formula.api as stm
X = np.append(np.ones(shape = (50,1)).astype(int), values = X,axis=1)