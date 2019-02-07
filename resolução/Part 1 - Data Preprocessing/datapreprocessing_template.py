# Data preprocessing

# Importing libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Creating variables 
X = dataset.iloc[:, :-1].values # Independant variables, Matrix of features
y = dataset.iloc[:,3].values # Dependant variable

# Split into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the independant variables (Feature scaling)
"""from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)"""
