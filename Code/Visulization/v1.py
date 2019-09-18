import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from pandas.plotting import andrews_curves


path = "../../Datasets/Data_Preprocessing/Data.csv"
dataset = pd.read_csv(path)


plt.figure()
andrews_curves(dataset,"Country")
plt.show()



# print(dataset.isnull().any().any())
# print(dataset.describe())
# print(dataset.head())
# print(dataset)
# X=dataset.iloc[:,[1,2,3]]
# y=dataset.iloc[:,4:].values
# print(X)
# # print(y)
# imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
# X[:,1:3]=imputer.fit_transform(X[:,1:3])

# df = {'Country':Y_test, 'pred': pred}
# data_frame= pd.DataFrame(data=df)


# andrews_curves()