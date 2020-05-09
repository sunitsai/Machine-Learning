# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:52:31 2020

@author: SUNIT JHA
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')


# To access the Dependent value from Dataset with 3 Col Country,Age,Salary in X
X = dataset.iloc[:,:-1].values

# To access the InDependent value from Dataset with 1 Col Purchased in Y
Y = dataset.iloc[:,-1].values





# Splitting the data into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)



# Training the Simple Regression model on Training Set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,Y_train)


# Predicting the Training Set Result
x_pred = regression.predict(X_train)


# Predicting the Test Set Result
y_pred = regression.predict(X_test)


# Visualing the Training Set
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,x_pred,color='blue')
plt.title("Salary Vs Experince (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()




# Visualing the Test Set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,x_pred,color='blue')
plt.title("Salary Vs Experince (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()




