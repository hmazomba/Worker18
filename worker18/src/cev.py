"""Cumulative explained variance gives the percentage of variance accounted for by each component. 
Explained Variance is used to measure the discreapancy between a model and actual data. 
This is the part of a model's total variance that is explained by factor and not due to error variance.""" 


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('../data/metal_concentration_results.csv')
#drop the first column
dataset.drop(dataset.iloc[:, 0:1], inplace=True, axis=1)

rows = len(dataset)
columns = len(dataset.columns)
print(columns)

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:0]

dataset = StandardScaler().fit_transform(dataset)
df = pd.DataFrame(dataset, columns=[y])

pca = PCA()
pca_model = pca.fit(df)
plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.axhline(y=0.8, color='r', linestyle='--', linewidth=1)
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
