# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:19:04 2021

@author: admin
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import the dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#handling missing data
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding categorical data
LabelEncoder_X = LabelEncoder()
X[:,0]= LabelEncoder_X.fit_transform(X[:,0])

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)

#splitting dataset into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.25,random_state = 0)

#feature scaling
scale_x = StandardScaler()
X_train = scale_x.fit_transform(X_train)
X_test = scale_x.transform(X_test)

