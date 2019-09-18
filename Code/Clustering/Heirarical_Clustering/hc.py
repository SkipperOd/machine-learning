import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv("../../../Datasets/Heirarical_Clustering/Mall_Customers.csv")

print(dataset.describe())

X= dataset.iloc[:,3].values
y= dataset.iloc[:,4].values


plt.scatter(X,y)
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.show()

# Data for dendogram
X = dataset.iloc[:,[3,4]].values
# Creating Dendogram
linkage = sch.linkage(X,method="ward")
dendogram= sch.dendrogram(linkage)
plt.xlabel("Clusters")
plt.ylabel("Eculidean Distance")
plt.show()


# Creating Agglomerative Clustering

aggloClustering = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_pred=aggloClustering.fit_predict(X)

print(y_pred)

# Plotting clusters
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='red',label="1")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='green',label="2")
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='blue',label="3")
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='purple',label="4")
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],s=100,c='cyan',label="5")
plt.legend()
plt.show()