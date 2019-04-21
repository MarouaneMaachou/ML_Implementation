#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 02:48:00 2019

@author: maachou
"""

import numpy as np
import scipy as sc
import scipy.stats  as stat


class PCA:
    
    
    def __init__(self,output_dim):
        self.output_dim=output_dim
        self.mean_training=None
        self.std_training=None
        
        
    def Scaler(self,X):
        nb_features=X.shape[1]
        self.mean_training=[None for i in range(nb_features)]
        self.std_training=[None for i in range(nb_features)] 
        for i in range(nb_features):
            mean=np.mean(X[:,i],axis=-1)
            st=np.std(X[:,i],axis=-1)
            X[:,i]=(X[:,i]-mean)*(1/st)
            self.mean_training[i]=mean
            self.std_training[i]=st
        return X
    def fit_transform(self,X):
        X=self.Scaler(X)
        self.cov=np.dot(np.transpose(X),X)
        eigen_values,eigen_vectors=np.linalg.eig(self.cov)
        order = np.argsort(eigen_values)[::-1]
        principal_components = eigen_vectors[:, order[:self.output_dim]]

        Z=np.dot(X,principal_components)
        return Z


