#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:21:39 2019

@author: maachou
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from Bayesian_linear import bayesian_linear_regression


Data=load_boston()
X=Data.data
Y=Data.target
X_train,X_test,y_train,y_test=train_test_split(X,Y)
Classifier=bayesian_linear_regression()
Classifier.fit(X_train,y_train)
print("test score:  ",Classifier.score(X_test,y_test))