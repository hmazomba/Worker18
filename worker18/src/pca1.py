import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('../data/metal_concentration_results.csv')
#drop the first column
dataset.drop(dataset.iloc[:, 0:1], inplace=True, axis=1)

X = dataset.iloc[:, 0:5].values
Y = dataset.iloc[:0]

dataset = StandardScaler().fit_transform(dataset)
df = pd.DataFrame(dataset, columns=[Y])

pca = PCA(n_components=2)
pca_model = pca.fit(df)
df_trans = pd.DataFrame(pca_model.transform(df), columns=['pca1', 'pca2'])