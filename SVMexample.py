from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Loading datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

#classes for diffferent species of iris
classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

# Splitting datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print('Predictions: ', prediction)
print('Accuracy: ', accuracy)
for i in range(len(prediction)):
    print(classes[prediction[i]])