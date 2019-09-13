import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
'''
Feature scaling is a method used to normalize the range of independent variables.
Since the range of values of raw data varies widely, it causes machine learning model to work properly with out feature scalling/normalization
many classifiers calculate the distance between two points by the Euclidean distance. 
If one of the features has a broad range of values, the distance will be governed by this particular feature.
Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

liberary to handle feature scalling of data
'''
from sklearn.preprocessing import StandardScaler
#Importing dataset
dataset = pd.read_csv("../../Datasets/Data_Preprocessing/Data.csv")
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1:].values

#Handeling missing data
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Handeling categorical data
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y[:,0] = labelencoder_Y.fit_transform(Y[:,0])

#spliting dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

'''
Feature scalling
'''
standardscaler_x = StandardScaler()
X_train = standardscaler_x.fit_transform(X_train)
X_test = standardscaler_x.transform(X_test)

print(X_train)
