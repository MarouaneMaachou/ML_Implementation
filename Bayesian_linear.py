#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:58:01 2019

@author: maachou
"""

import numpy as np
import scipy as sc
import scipy.stats  as stat
def sigmoid(z):
    return 1/(1+np.exp(-z))
"""def initialize_prior(n_parameters):
        mean=np.zeros((n_parameters,))
        variance=np.identity(n_parameters)
        def pdf(x):
            return stat.multivariate_normal.pdf(x,mean=mean,cov=variance)
        return pdf
def likelihood(X,Y,theta):
    output_dim=Y.shape[1]
    n_examples,n_dimensions=X.shape[0],X.shape[1]
    n_parameters=n_dimensions
    variance=np.identity(output_dim)
    mean=np.dot(theta,X)
    return stat.multivariate_normal.pdf(Y,mean=mean,cov=variance)
def pdf_likelihood(X,Y):
    def pdf(theta):
        return likelihood(X,Y,theta)
    return pdf


def posterior(X,Y,theta):
    likelihood=likelihood(X,Y,theta)"""
    
def fit_params(X,Y,prior_mean,prior_variance):
    output_dim=Y.shape[1]
    model_variance=np.identity(output_dim)
    posterior_mean=np.dot(np.linalg.inv(np.dot(np.transpose(X),X)+np.linalg.inv(prior_variance)),np.dot(np.linalg.inv(prior_variance),prior_mean)+np.dot(np.transpose(X),Y))
    posterior_variance=np.linalg.inv(np.dot(np.transpose(X),X)+np.linalg.inv(prior_variance))
    return posterior_mean,posterior_variance
class  bayesian_linear_regression:
    def __init__(self,prior_sigma=1000,prediction_samples=10000,max_iter=1000):
        self.prior_sigma=prior_sigma
        self.posterior_mean=None
        self.samples=prediction_samples
        self.max_iter=max_iter
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
    def fit(self,X,Y):
        Y=np.reshape(Y,(-1,1))
        X=self.Scaler(X)
        ones=np.ones((X.shape[0],1))
        X_bias=np.concatenate((ones,X),axis=1)
        n_parameters=X_bias.shape[1]

        self.posterior_mean=np.zeros((n_parameters,1))
        self.posterior_variance=self.prior_sigma*np.identity(n_parameters)
        for i in range(self.max_iter):
            #print(i)
            self.posterior_mean,self.posterior_variance=fit_params(X_bias,Y,self.posterior_mean,self.posterior_variance)
        
    def predict(self,X):
        for i in range(X.shape[1]):
            mean=self.mean_training[i]
            st=self.std_training[i]
            X[:,i]=(X[:,i]-mean)*(1/st)

        ones=np.ones((X.shape[0],1))
        X_bias=np.concatenate((ones,X),axis=1)
        sample_predictions=[]
        self.posterior_mean=self.posterior_mean.reshape((-1,))
        for i in range(self.samples):
            theta=np.random.multivariate_normal(self.posterior_mean,self.posterior_variance)
            sample_predictions.append(np.dot(X_bias,theta))
            
        predictions=np.transpose(np.array(sample_predictions))

        
        return np.mean(predictions,axis=1),X_bias
    def score(self,X,Y):
        Y=Y.reshape(-1,1)
        y,X_bias=self.predict(X)
        y=y.reshape(-1,1)
        difference=Y-y
        mse=np.mean(np.multiply(difference,difference),axis=0)
        return mse[0]





    
            
            
        
    
    
    
    
    

        