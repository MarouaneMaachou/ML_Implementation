#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:12:23 2019

@author: maachou
"""

import numpy as np
import scipy as sc
import scipy.stats  as stat









class GMM():
    def __init__(self,K=2,max_iter=100,eps=0.0000001):
        self.k=K
        self.mean_training=None
        self.std_training=None
        self.alpha=np.random.rand(self.k)
        self.parameters=[]
        self.max_iter=max_iter
        self.eps=eps

    def Scaler(self,X):
        nb_features=X.shape[1]
        for i in range(nb_features):
            mean=np.mean(X[:,i],axis=-1)
            st=np.std(X[:,i],axis=-1)
            X[:,i]=(X[:,i]-mean)*(1/st)
            self.mean_training[i]=mean
            self.std_training[i]=st
    def EM(self,X):
        
        membership=np.zeros((X.shape[0],self.k))
        for step in range(self.max_iter):
            print("step:",step)
            #The E step
            for i in range(X.shape[0]):
                normalisation=0
                for k in range(self.k):
    
                    membership[i][k]=self.alpha[k]*stat.multivariate_normal.pdf(X[i],mean=self.parameters[k]["mean"],cov=self.parameters[k]["variance"])
                    normalisation+=membership[i][k]
                membership[i]=membership[i]/normalisation
    
                
            #The M STEP
            N=np.zeros((self.k))
            for k in range(self.k):
                for i in range(X.shape[0]):
                    N[k]+=membership[i][k]
            self.alpha=N/X.shape[0]
            for k in range(self.k):
                self.parameters[k]["mean"]=np.zeros(X.shape[1])
                for i in range(X.shape[0]):
                    self.parameters[k]["mean"]+=(1/N[k])*membership[i][k]*X[i]
                
                self.parameters[k]["variance"]=np.zeros((X.shape[1],X.shape[1]))
                for i in range(X.shape[0]):
    
                    value=(X[i]-self.parameters[k]["mean"]).reshape(-1,1)
                    empirical_cov=np.dot(value,np.transpose(value))
                    self.parameters[k]["variance"]+=(1/N[k])*membership[i][k]*empirical_cov+self.eps*np.identity(X.shape[1])
                
        return None       
    def fit(self,X):
        n_features=X.shape[1]
        n_examples=X.shape[0]
        
        for i in range(self.k):
            self.parameters.append({"mean":np.random.rand(n_features),"variance":(1/(n_examples**2))*np.dot(np.transpose(X),X)})
        self.EM(X)
        return None
    def Means(self):
        for k in range(self.k):
            print("mean of cluster ",k,"is : ",self.parameters[k]['mean'])

    def predict(self,X):
        Y=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            min_distance=10000000000
            for k in range(self.k):
                if np.linalg.norm(X[i]-self.parameters[k]["mean"])<=min_distance:
                    min_distance=np.linalg.norm(X[i]-self.parameters[k]["mean"])
                    Y[i]=k
        return Y
                    
                
