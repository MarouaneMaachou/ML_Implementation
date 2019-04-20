#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:18:30 2019

@author: maachou
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from linear import logistic_regression


Data=load_breast_cancer()
X=Data.data
Y=Data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y)
Classifier=logistic_regression()
Classifier.fit(X_train,y_train)
print("test score:  ",Classifier.score(X_test,y_test))