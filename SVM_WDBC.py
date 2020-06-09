

import numpy as np
import pandas as pd
from sklearn import svm 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


df = pd.read_csv('wdbc.data',header=None)


df[1] = np.where(df[1] == 'B', 0 , 1)

array = df.values

X = array[:,2:31]
Y = array [:,1]



clf = svm.SVC(kernel = 'linear', C=1, gamma='auto')

kf = KFold(n_splits=10)
accuracy = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index],X[test_index]
    y_train, y_test = Y[train_index],Y[test_index]
    clf.fit(X_train,y_train)
    prediction = clf.predict(X_test)
    print(accuracy_score(y_test,prediction))
    accuracy.append(accuracy_score(y_test,prediction))


clf = svm.SVC(kernel = 'rbf', gamma='auto')

kf = KFold(n_splits=10)
accuracy = []

for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index],X[test_index]
	y_train, y_test = Y[train_index],Y[test_index]
	clf.fit(X_train,y_train)
	prediction = clf.predict(X_test)
	print(accuracy_score(y_test,prediction))
	accuracy.append(accuracy_score(y_test,prediction))
 
 
print(accuracy) 
print("Mean Accuracy: ",np.mean(accuracy))


clf = svm.SVC(kernel = 'poly', degree=2, gamma='auto')

kf = KFold(n_splits=10)
accuracy = []

for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index],X[test_index]
	y_train, y_test = Y[train_index],Y[test_index]
	clf.fit(X_train,y_train)
	prediction = clf.predict(X_test)
	print(accuracy_score(y_test,prediction))
	accuracy.append(accuracy_score(y_test,prediction))
 
 
print(accuracy) 
print("Mean Accuracy: ",np.mean(accuracy))
