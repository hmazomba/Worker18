
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

X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:0]

dataset = StandardScaler().fit_transform(dataset)
df = pd.DataFrame(dataset, columns=[Y])

pca = PCA(n_components=2)
pca_model = pca.fit(df)
df_trans = pd.DataFrame(pca_model.transform(df), columns=['pca1', 'pca2'])
rng = np.random.RandomState(0)
colors = rng.rand(5)
sizes = 1000 * rng.rand(5)
scatter_projection = plt.scatter(df_trans['pca1'], df_trans['pca2'], c=colors, s=sizes, alpha=0.8)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar()
plt.show()