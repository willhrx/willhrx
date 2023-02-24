# My first attempt at machine learning 

import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

wine = sklearn.datasets.load_wine()

# Wine had to be chosen for my dad

X_train, X_test, Y_train, Y_test = train_test_split(wine.data,wine.target)

# Made some training and testing sets boom

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Accuracy tends to be between 91% and 95%





