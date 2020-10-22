"""Explained Variance is used to measure the discreapancy between a model and actual data. 
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


X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:0]

dataset = StandardScaler().fit_transform(dataset)
df = pd.DataFrame(dataset, columns=[y])

pca = PCA()
pca_model = pca.fit(df)

plt.plot(pca_model.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.show()