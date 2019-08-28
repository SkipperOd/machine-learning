import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#importing data
dataset = pd.read_csv("../../Datasets/Polinomial_Regression/Position_Salaries.csv")
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1:].values

print(X.shape)
print(Y.shape)

#Handling categorical data
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
#whoever python liberary for linear regression is taking care of dummy variable trap. 
# X=X[:,1:]

#Splitting of data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Model 
regression = LinearRegression()
