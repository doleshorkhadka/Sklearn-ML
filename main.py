from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

#fetching datasets and split into features and labels

irish = datasets.load_iris()

X = irish.data
y = irish.target
print(X.shape)
print(y.shape)

# train datasets and test datasets
X_train , X_test , y_train,y_test = train_test_split(X,y ,test_size=0.3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)