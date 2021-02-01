# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:08:54 2021

@author: admin
"""


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

#import the dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#splitting dataset into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=1/3,random_state = 0)


#fit simple linear regression to training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting test set results
y_pred = regressor.predict(X_test)

#visualize training set
plt.scatter(X_train, Y_train,color='r')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.title('Salary-Experience(Training set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

#visualize test set
plt.scatter(X_test, Y_test,color='r')
plt.plot(X_train, regressor.predict(X_train) , color='b')
plt.title('Salary-Experience(Test set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()