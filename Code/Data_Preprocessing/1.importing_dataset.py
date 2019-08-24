'''
importing basic liberaries 
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
importing data preprocessing dataset
'''

dataset = pd.read_csv("../../Datasets/Data_Preprocessing/Data.csv")

'''
Split up dataset in independent and dependent variables. 
'''
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1:].values

'''
By using property values
we will get a features in the form of lists

and we dont use values property we will get data frame
Exmpale: "dataset.iloc[:,:-1]"
'''
