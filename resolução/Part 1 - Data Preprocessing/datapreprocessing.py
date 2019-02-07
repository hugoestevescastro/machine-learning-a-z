# Data preprocessing

# Importing libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Creating variables 
# object.iloc[lines, columns]
X = dataset.iloc[:, :-1].values # Independant variables, Matrix of features
y = dataset.iloc[:,3].values # Dependant variable

# Handle missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

# Nationality to numeric

# É necessário fazer um dummy column para o algoritmo não 
# achar que por terem valores númerico diferentes, espanha > alemanha etc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()

# Purchased to tinyint
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)


# Split into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the independant variables (Feature scaling)
# Se fosse regressão,isto teria de ser feito para as variáveis dependentes
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)