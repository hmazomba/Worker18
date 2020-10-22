import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix

dataset = pd.read_csv('../data/metal_concentration_results.csv')
#drop the first column
dataset.drop(dataset.iloc[:, 0:1], inplace=True, axis=1)

tc = dataset.corr()
sns.heatmap(tc)
plt.show()