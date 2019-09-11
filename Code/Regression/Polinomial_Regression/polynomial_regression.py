import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#importing data
dataset = pd.read_csv("../../../Datasets/Polinomial_Regression/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y= dataset.iloc[:,-1:].values

print(X.shape)
print(Y.shape)

#Handling categorical data
# labelencoder_X = LabelEncoder()
# X[:,0] = labelencoder_X.fit_transform(X[:,0])
# onehotencoder = OneHotEncoder(categorical_features=[0])
# X=onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
#whoever python liberary for linear regression is taking care of dummy variable trap. 
# X=X[:,1:]

#Splitting of data 
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Model 
regression = LinearRegression()

regression.fit(X,Y)
print(regression.score(X,Y))
pred =regression.predict(X)

poly_reg = PolynomialFeatures(3)

X_poly = poly_reg.fit_transform(X)
print(X_poly.shape)

reg_2 = LinearRegression()
reg_2.fit(X_poly,Y)
print(reg_2.score(X_poly,Y))
print(reg_2.predict(poly_reg.fit_transform(6.5)))
pred_2=reg_2.predict(X_poly)

plt.scatter(X,Y)
plt.plot(X,pred,color="red")
plt.plot(X,pred_2,color="green")
# plt.scatter(,6.5,color="purple")
plt.show()