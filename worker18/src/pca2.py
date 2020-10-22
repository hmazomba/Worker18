import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from pca import pca


dataset = pd.read_csv('../data/metal_concentration_results.csv')
#drop the first column
dataset.drop(dataset.iloc[:, 0:1], inplace=True, axis=1)

X = dataset.iloc[:, 0:5].values
X = pd.DataFrame(data=X, columns=['As','Mn','Pb', 'Zn', 'Fe'])
#Y = dataset.iloc[:0]


model = pca(n_components=3)
results = model.fit_transform(X)

ax = model.biplot3d(n_feat=5, legend=True)


