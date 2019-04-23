#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:31:22 2019

@author: maachou
"""

import scipy as sc
import scipy.stats  as stat
import numpy as np
def distance(u,v):
    return np.linalg.norm(u-v)
def nearest(u,X,k):
    neirest=np.zeros((k,X.shape[1]))
    Distances=[distance(u,v) for v in X]
    neirest=np.argsort(Distances)[:k]
    
    return neirest



class LWR:
    def __init__(self,brandwidth=10e10):
        self.brandwidth=brandwidth
        self.X_train=None
        self.Y_train=None
       
    def fit(self,X,Y):
        self.X_train=X
        self.Y_train=Y
        
    def predict(self,X):
        eps=10e-20
        weights=np.zeros((X.shape[0],self.X_train.shape[0]))
        Y_pred=np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            for j in range(self.X_train.shape[0]):
                weights[i][j]=np.exp(-np.linalg.norm(X[i]-self.X_train[j])/self.brandwidth)
            normalisation=np.sum(weights[i])
            mu_x=(np.sum(weights[i].reshape((-1,1))*self.X_train,axis=0)/normalisation)
            mu_y=np.sum(weights[i]*self.Y_train)/normalisation
            sigma_x=np.dot((weights[i].reshape((-1,1))*(self.X_train-mu_x)).T,self.X_train-mu_x)/normalisation
            sigma_x_y=np.dot((weights[i].reshape((-1,1))*(self.X_train-mu_x)).T,self.Y_train-mu_y)/normalisation
            Y_pred[i]=mu_y+np.dot(np.dot(sigma_x_y,np.linalg.inv(sigma_x+eps*np.identity(sigma_x.shape[0]))),(X[i]-mu_x))
            
        return Y_pred
            
            
        
    def score(self,X,Y):
        Y=Y.reshape(-1,1)
        y=self.predict(X)
        difference=Y-y
        mse=np.mean(np.multiply(difference,difference),axis=0)
        return mse


