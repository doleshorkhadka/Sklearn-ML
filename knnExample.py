from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

dataset = pd.read_csv('car.data')

X = dataset[[
    'buying',
    'maint',
    'safety'
]].values
y = dataset[[
    'class'
]]

# Converting X using label encorder
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

# Converting y
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}
y['class'] = y['class'].map(label_mapping)

# build the model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print('Prediction: ', prediction)
print('Accuracy: ', accuracy)
print('Actual value: ', y.values[20])
print('Predicted value: ', knn.predict(X)[20])
