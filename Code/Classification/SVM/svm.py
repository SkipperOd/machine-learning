import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
# importing data
dataset = pd.read_csv("../../../Datasets/SVM/Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
# X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 4].values
print(dataset.describe())

# splitting of data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# if needs be bellow code descibes how to handle feature scalling, Some models already apply this so we dont need to do it for every model.
# poly_reg = PolynomialFeatures(3)
# X_poly = poly_reg.fit_transform(X)
# print(X_poly.shape)
standardscaler_x = StandardScaler()
X_train = standardscaler_x.fit_transform(X_train)
X_test = standardscaler_x.transform(X_test)


classifier = SVC(kernel="poly", random_state=0)
classifier.fit(X_train, Y_train)

score = classifier.score(X_test, Y_test)
print("Classifier score: ", score)

pred = classifier.predict(X_test)

# making confusion matrix
cm = confusion_matrix(Y_test, pred)
df = {'Y_test':Y_test, 'pred': pred}
data_frame= pd.DataFrame(data=df)
data_frame.to_csv("../../Datasets/SVM/comparison.csv")
print(data_frame)
print(cm)


def plotting(X, Y):
    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X, Y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Logistic Regression')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


plotting(X_train, Y_train)
plotting(X_test, Y_test)
