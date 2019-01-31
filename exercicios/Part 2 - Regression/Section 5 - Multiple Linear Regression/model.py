# Multiple Linear Regression

# !-Data Preprocessing
import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values # Independant variables
y = dataset.iloc[:,4].values # Dependant variable (target)

# Dummy variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])
one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# !-Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# !-Predicting the test set results
y_pred = regressor.predict(X_test)

# !-Building the optimal model using Backward Elimination

# Uma vez que a lib statsmodels nao considera o b0 na formula 
# y = b0 + b1*x1 + bn*xn,
# É necessário adicionar ao array de X, no entanto o regressor considera o b0
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
sl = 0.05 # Significance level
X_Modeled = backwardElimination(X_opt, sl)