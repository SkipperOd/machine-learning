import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

#importing data
dataset = pd.read_csv("")
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1:].values

#If needs be bellow code describes how to handle missing data
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#if needs be bellow code describes how to handle categorical data.
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y[:,0] = labelencoder_Y.fit_transform(Y[:,0])

#splitting of data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#if needs be bellow code descibes how to handle feature scalling, Some models already apply this so we dont need to do it for every model.
standardscaler_x = StandardScaler()
X_train = standardscaler_x.fit_transform(X_train)
X_test = standardscaler_x.transform(X_test)