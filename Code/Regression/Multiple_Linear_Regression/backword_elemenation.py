import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#importing data
dataset = pd.read_csv("../../../Datasets/Multiple_Linear_Regression/50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1:].values

print(X.shape)
print(Y.shape)

#Handling categorical data
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
#whoever python liberary for linear regression is taking care of dummy variable trap. 
X=X[:,1:]

#Splitting of data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Model 
regression = LinearRegression()
regression.fit(X_train,Y_train)
score = regression.score(X_test,Y_test) * 100
print("Model score:",score )
y_pred=regression.predict(X_test)


# print(X_train)

# plt.scatter(X_train,Y_train)
# plt.plot(X_test,y_pred,color="r")
# plt.scatter(X_test,y_pred,color="g")
# plt.show()

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:,[0,1,2,3,4]]
regression_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:,[0,1,3,4]]
regression_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
# print(regression_OLS.summary())


X_opt = X[:,[0,3,4]]
regression_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
# print(regression_OLS.summary())

# print(X_opt.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_opt,Y,test_size=0.2,random_state=0)

regression.fit(X_train,Y_train)
print(regression.score(X_test,Y_test) * 100)


