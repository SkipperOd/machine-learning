import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
'''
Data consisting of salary with relation to number of experience in years
'''
#Importing dataset and converting it in X and Y
dataset = pd.read_csv("../../Datasets/Simple_Liner_Regression/Salary_Data.csv")
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values
print(X.shape)
print(Y.shape)
#Splitting data in to testing and traning set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


#We dont need to apply feature scalling on linear regression model as the liberery will do it for us

#creating linear regression class 
regression = LinearRegression()
regression.fit(X_train,Y_train)
#how well our model is trained 
score = regression.score(X_test,Y_test) * 100

print("Model score:",score )

#predicting value
y_pred=regression.predict(X_test)

# print("X Train",Y_test)
# print("Y Prediction",y_pred)

#For visualizing above result we use matplot lib

plt.scatter(X_train,Y_train)
plt.plot(X_test,y_pred,color="r")
plt.scatter(X_test,y_pred,color="g")
plt.show()








