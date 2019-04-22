#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:20:17 2019

@author: maachou
"""

import scipy as sc
import scipy.stats  as stat
import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))
def distance(u,v):
    return np.linalg.norm(u-v)
def neirest(u,X,k):
    neirest=np.zeros((k,X.shape[1]))
    Distances=[distance(u,v) for v in X]
    neirest=np.argsort(Distances)[:k]
    
    return neirest
        
        
    

class KNN:
    def __init__(self,n_neighbors):
        self.n_neighbors=n_neighbors
        self.X_train=None
        self.Y_train=None
       
    def fit(self,X,Y):
        self.X_train=X
        self.Y_train=Y
        print(Y)
    
    def predict(self,X):
        Y=np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            choice=np.argmax(np.bincount(self.Y_train[neirest(X[i],self.X_train,self.n_neighbors)]))
            Y[i]=choice
            
        return Y
    def score(self,X,Y):
        nombre_total=X.shape[0]
        Y=Y.reshape(-1,1)
        y_pred=self.predict(X)
        nombre_juste=np.sum(np.array(Y==y_pred,dtype=int))
        score=nombre_juste/nombre_total
        return score


