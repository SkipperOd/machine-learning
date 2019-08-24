import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

'''
As machine learning models are based of mathematical equations.
It is quite obvious it will cause some issues so in order to handle categorical data
we will encode text in to numbers.

liberary to handle encoding of text to number
'''
from sklearn.preprocessing import LabelEncoder

#Importing dataset
dataset = pd.read_csv("../../Datasets/Data_Preprocessing/Data.csv")
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1:].values
#Handeling missing data
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
'''
encoding label to number can be done easily. 
'''
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

'''
There is a problem. 
Its good that we have changed categorical values to number. 
But the problem remains the same. 
Machine leanring model i.e. mathematical equations will think
of ranking up rows as per increasing value.  
so in order to solve this problem we will use dummy variables. 
To use dummy variables we will use anther class known as "ONEHOTENCODER"
'''
from sklearn.preprocessing import OneHotEncoder
'''
and again we will fit and transform encoder as per coloumn in need
'''
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
print(X)
labelencoder_Y = LabelEncoder()
Y[:,0] = labelencoder_Y.fit_transform(Y[:,0])

print(Y)