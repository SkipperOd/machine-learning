import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv("../../../Datasets/K-Means/Mall_Customers.csv")

print(dataset.describe())

X = dataset.iloc[:,3].values
y= dataset.iloc[:,4].values

plt.scatter(X,y)
plt.show()


X = dataset.iloc[:,[3,4]].values
# finding optimal number of clusters for k mean by using elbow method
wcss=[]
for i in range(1,11):
      # print(i)
      kmean = KMeans(n_clusters=i,n_init=10,max_iter=300,random_state=0)
      kmean.fit(X)
      wcss.append(kmean.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("Number of clusters")
plt.ylabel("Wcss")
plt.show()



#training model with correct number of clusters now. 
kmean = KMeans(n_clusters=5,n_init=10,max_iter=300,random_state=0)
y_pred=kmean.fit_predict(X)
# kmean.predict
print(y_pred)

# Plotting clusters


plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='red',label="1")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='green',label="2")
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='blue',label="3")
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='purple',label="4")
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],s=100,c='cyan',label="5")
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s=300,c="yellow",label="centroid")
plt.legend()
plt.show()
