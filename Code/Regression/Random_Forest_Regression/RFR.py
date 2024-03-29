'''
Keep in mind decision tree is non linear and non contenious model 

'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


dataset = pd.read_csv("../../../Datasets/Decision_Tree_Regression/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y= dataset.iloc[:,-1:].values


regression = RandomForestRegressor(n_estimators=300,random_state=0)

regression.fit(X,Y)


pred= regression.predict(X)

y_pred = regression.predict(6.5)
#visualising the decision tree regression results as they are non linear and non contineous 
x_grid= np.arange(min(X),max(X),0.01)
x_grid= x_grid.reshape((len(x_grid),1))
plt.scatter(X,Y)
plt.scatter(6.5,y_pred,color='red')
plt.plot(X,pred,color="purple")
plt.plot(x_grid,regression.predict(x_grid),color="black")
plt.show()



print(regression.predict(6.5))
