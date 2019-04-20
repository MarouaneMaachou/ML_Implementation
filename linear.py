#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:36:40 2019

@author: maachou
"""


import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))





class  linear_regression:
    def __init__(self,max_iter=100,learning_rate=0.1):
        self.weights=None
        self.max_iter=max_iter
        self.learning_rate=learning_rate
    def Scaler(self,X):
        nb_features=X.shape[1]
        for i in range(nb_features):
            mean=np.mean(X[:,i],axis=-1)
            st=np.std(X[:,i],axis=-1)
            X[:,i]=(X[:,i]-mean)*(1/st)
        return X
        
        return None
    def predict(self,X):
        X=self.Scaler(X)
        ones=np.ones((X.shape[0],1))
        X_bias=np.concatenate((ones,X),axis=1)
        if self.weights.any()==None:
            print("error : Weights not initialized")
            return None
        else:
            y=np.matmul(X_bias,np.transpose(self.weights))
            
        return y,X_bias
    def fit(self,X,Y):
        
        Y=Y.reshape(-1,1)
        nb_examples,nb_features=X.shape[0],X.shape[1]
        self.weights=np.random.normal(0,0.01,(1,nb_features+1))
        for iteration in range(self.max_iter):
            y,X_bias=self.predict(X)
           
            difference=Y-y
            mse=np.mean(np.multiply(difference,difference),axis=0)
            gradient=(1/nb_examples)*np.matmul(np.transpose(y-Y),X_bias)
            self.weights=self.weights-self.learning_rate*gradient
            print("epoch " ,iteration,"  MSE: ",mse)
    def score(self,X,Y):
        Y=Y.reshape(-1,1)
        y,X_bias=self.predict(X)
        difference=Y-y
        mse=np.mean(np.multiply(difference,difference),axis=0)
        return mse
    
    
    
class  regularized_linear_regression:
    def __init__(self,max_iter=100,learning_rate=0.1,landa=0.001):
        self.weights=None
        self.max_iter=max_iter
        self.learning_rate=learning_rate
        self.landa=landa
    def Scaler(self,X):
        nb_features=X.shape[1]
        for i in range(nb_features):
            mean=np.mean(X[:,i],axis=-1)
            st=np.std(X[:,i],axis=-1)
            X[:,i]=(X[:,i]-mean)*(1/st)
        return X
        
        return None
    def predict(self,X):
        X=self.Scaler(X)
        ones=np.ones((X.shape[0],1))
        X_bias=np.concatenate((ones,X),axis=1)
        if self.weights.any()==None:
            print("error : Weights not initialized")
            return None
        else:
            y=np.matmul(X_bias,np.transpose(self.weights))
            
        return y,X_bias
    def fit(self,X,Y):
        
        Y=Y.reshape(-1,1)
        nb_examples,nb_features=X.shape[0],X.shape[1]
        self.weights=np.random.normal(0,0.01,(1,nb_features+1))
        for iteration in range(self.max_iter):
            y,X_bias=self.predict(X)
           
            difference=Y-y
            mse_regularized=np.mean(np.multiply(difference,difference),axis=0)+self.landa*np.sum(np.multiply(self.weights,self.weights))
            gradient=(1/nb_examples)*(np.matmul(np.transpose(y-Y),X_bias)+self.landa*self.weights)
            self.weights=self.weights-self.learning_rate*gradient
            print("epoch " ,iteration,"  MSE: ",mse_regularized)
    def score(self,X,Y):
        Y=Y.reshape(-1,1)
        y,X_bias=self.predict(X)
        difference=Y-y
        mse_regularized=np.mean(np.multiply(difference,difference),axis=0)+self.landa*np.sum(np.multiply(self.weights,self.weights))
        return mse_regularized    
    





class logistic_regression:
    def __init__(self,max_iter=100,learning_rate=0.1):
        self.weights=None
        self.max_iter=max_iter
        self.learning_rate=learning_rate
    def Scaler(self,X):
        nb_features=X.shape[1]
        for i in range(nb_features):
            mean=np.mean(X[:,i],axis=-1)
            st=np.std(X[:,i],axis=-1)
            X[:,i]=(X[:,i]-mean)*(1/st)
        return X
        

    def predict(self,X):
        X=self.Scaler(X)
        ones=np.ones((X.shape[0],1))
        X_bias=np.concatenate((ones,X),axis=1)
        if self.weights.any()==None:
            print("error : Weights not initialized")
            return None
        else:
            y_hat=sigmoid(np.matmul(X_bias,np.transpose(self.weights)))
            y_pred=np.array(y_hat>0.5,dtype=int)
            
            
        return y_hat,y_pred,X_bias
    def fit(self,X,Y):
        Y=Y.reshape(-1,1)
        nb_examples,nb_features=X.shape[0],X.shape[1]
        self.weights=np.random.normal(0,0.01,(1,nb_features+1))
        for iteration in range(self.max_iter):
            y_hat,y_pred,X_bias=self.predict(X)
            log_error=-np.multiply(Y,np.log(y_hat))
            cross_entropy=np.mean(log_error,axis=0)
            gradient=(1/nb_examples)*np.matmul(np.transpose(y_hat-Y),X_bias)
            self.weights=self.weights-self.learning_rate*gradient
            print("epoch " ,iteration,"  Cross_entropy loss: ",cross_entropy)
            print("epoch " ,iteration,"  Accuracy: ",self.score(X,Y))
    def score(self,X,Y):
        nombre_total=X.shape[0]
        Y=Y.reshape(-1,1)
        
        y_hat,y_pred,X_bias=self.predict(X)
        nombre_juste=np.sum(np.array(Y==y_pred,dtype=int))
        score=nombre_juste/nombre_total
        return score

class regularized_logistic_regression:
    def __init__(self,max_iter=100,learning_rate=0.1,landa=0.01):
        self.weights=None
        self.max_iter=max_iter
        self.learning_rate=learning_rate
        self.landa=landa
    def Scaler(self,X):
        nb_features=X.shape[1]
        for i in range(nb_features):
            mean=np.mean(X[:,i],axis=-1)
            st=np.std(X[:,i],axis=-1)
            X[:,i]=(X[:,i]-mean)*(1/st)
        return X
        

    def predict(self,X):
        X=self.Scaler(X)
        ones=np.ones((X.shape[0],1))
        X_bias=np.concatenate((ones,X),axis=1)
        if self.weights.any()==None:
            print("error : Weights not initialized")
            return None
        else:
            y_hat=sigmoid(np.matmul(X_bias,np.transpose(self.weights)))
            y_pred=np.array(y_hat>0.5,dtype=int)
            
            
        return y_hat,y_pred,X_bias
    def fit(self,X,Y):
        Y=Y.reshape(-1,1)
        nb_examples,nb_features=X.shape[0],X.shape[1]
        self.weights=np.random.normal(0,0.01,(1,nb_features+1))
        for iteration in range(self.max_iter):
            y_hat,y_pred,X_bias=self.predict(X)
            log_error=-np.multiply(Y,np.log(y_hat))
            print(log_error.shape)
            cross_entropy_regularized=np.mean(log_error,axis=0)+self.landa*np.sum(np.multiply(self.weights,self.weights))
            gradient=(1/nb_examples)*(np.matmul(np.transpose(y_hat-Y),X_bias)+self.landa*self.weights)
            self.weights=self.weights-self.learning_rate*gradient
            print("epoch " ,iteration,"  Cross_entropy loss: ",cross_entropy_regularized)
            print("epoch " ,iteration,"  Accuracy: ",self.score(X,Y))
    def score(self,X,Y):
        nombre_total=X.shape[0]
        Y=Y.reshape(-1,1)
        
        y_hat,y_pred,X_bias=self.predict(X)
        nombre_juste=np.sum(np.array(Y==y_pred,dtype=int))
        score=nombre_juste/nombre_total
        return score

        
            
        

