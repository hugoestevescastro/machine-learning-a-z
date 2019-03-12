# Apriori

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Data preparation
# Transaction must be a list of lists [[...], [...], ...]
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training Apriori on the dataset

from apyori import apriori
# min_support -> produto comprado 3 vezes por dia x 7 dias (1semana) / total de transações (3*7/7500)
# min_confidence -> try/error até fazer sentido
# min_lift -> try/error até fazer sentido
# min_length -> numero minimo de produtos ligados
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
