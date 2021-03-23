# importing libraries
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.cluster import AgglomerativeClustering

# loading data
path = r"G:\rauf\STEPBYSTEP\Data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array=data.values
X = array[:,0:8]
y = array[:,8]
print(data.shape)
print(data.head())

#splitting data
patient_data = data.iloc[:,3:5].values

#create model
AClus = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

# train model
AClus.fit_predict(patient_data)