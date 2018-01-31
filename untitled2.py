# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:34:49 2018

@author: laksh
"""

# Data Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('winequalityred.csv')
X = dataset.iloc[:, :11].values
y = dataset.iloc[:, 11].values

#Train and test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

y_pred1 = classifier1.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
62.8


#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, y_train)

y_pred2 = classifier2.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred2)
48

#SVM
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear' , random_state = 0)
classifier3.fit(X_train, y_train)

y_pred3 = classifier3.predict(X_test)

cm3 = confusion_matrix(y_test, y_pred3)
63

#KernelSVM
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf' , random_state = 0)
classifier4.fit(X_train, y_train)

y_pred4 = classifier4.predict(X_test)

cm4 = confusion_matrix(y_test, y_pred4)
62

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train)

y_pred5 = classifier5.predict(X_test)

cm5 = confusion_matrix(y_test, y_pred5)
53

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier6.fit(X_train, y_train)

y_pred6 = classifier6.predict(X_test)

cm6 = confusion_matrix(y_test, y_pred6)
62.5

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier7 = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier7.fit(X_train, y_train)

y_pred7 = classifier7.predict(X_test)

cm7 = confusion_matrix(y_test, y_pred7)
67

#k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier7, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'criterion':['gini']},
               {'criterion':['entropy']}]
                
gridsearch = GridSearchCV(estimator = classifier7,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10,
                          n_jobs = -1)

gridsearch = gridsearch.fit(X_train, y_train)

best_accuracy = gridsearch.best_score_
best_parameters = gridsearch.best_params_










