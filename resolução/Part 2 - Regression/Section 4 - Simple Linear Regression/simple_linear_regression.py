# Simple Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Creating variables 
X = dataset.iloc[:, :-1].values # Independant variables, Matrix of features
y = dataset.iloc[:,1].values # Dependant variable

# Split into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Building a plot of the training set results
plt.scatter(X_train, y_train, color='red') # Observation points
plt.plot(X_train, regressor.predict(X_train), color='blue') # Draw the regression line
plt.title('Salary vs Experience (Training set results)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Building a plot of the testing set results
plt.scatter(X_test, y_test, color='red') # Observation points
plt.plot(X_train, regressor.predict(X_train), color='blue') # Draw the regression line
plt.title('Salary vs Experience (Test set results)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
