import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
'''
we need to split data in to traning and testing data so that we can see
if our model is being trained correctly or not. 

liberary to handle spliting of data
'''
from sklearn.model_selection import train_test_split


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

'''
spliting of dataset
'''

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)