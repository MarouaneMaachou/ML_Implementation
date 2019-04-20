#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:31:08 2019

@author: maachou
"""


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from linear import regularized_linear_regression


Data=load_boston()
X=Data.data
Y=Data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
Classifier=regularized_linear_regression(landa=0.01)
Classifier.fit(X_train,y_train)
print("test score:  ",Classifier.score(X_test,y_test))