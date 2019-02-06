# Polynomial regression

# Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting the regression model to the dataset
## NEED TO CREATE THE REGRESSOR

# Visualising the Regression Model results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Regression Model
y_pred = regressor.predict(np.array(6.5).reshape(1, -1))