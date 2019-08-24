import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Missing data can lead to issue in machine learning models as they are mathematical equations so in order to solve that issue 
we replace missing values in a number of ways.  
liberary to handle missing data
"""
from sklearn.preprocessing import Imputer

#Importing dataset
dataset = pd.read_csv("../../Datasets/Data_Preprocessing/Data.csv")
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1:].values

'''
First method is to "REMOVE" line with missing data. 
But that can be dangerous condering if data set have crucial data.
Or
Common method to handle this problem is to take "MEAN" of the colomn.
To take mean we will use "IMPUTER" and its predefined functions.   
'''
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)



