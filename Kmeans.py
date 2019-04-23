#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:18:06 2019

@author: maachou
"""

import scipy as sc
import scipy.stats  as stat
import numpy as np
def distance(u,v):
    return np.linalg.norm(u-v)

def nearest(u,X,k):
    nearest=np.zeros((k,X.shape[1]))
    Distances=[distance(u,v) for v in X]
    nearest=np.argsort(Distances)[:k]
    
    return nearest
def centre(U):
    center=np.zeros((1,U.shape[1]))
    for i in range(U.shape[0]):
        center+=U[i]
    return center/U.shape[0]

class Kmeans:
    def __init__(self,nb_clusters,max_iter=500,eps=0.001):
        self.nb_clusters=nb_clusters
        self.centroids=None
        self.max_iter=max_iter
        self.eps=eps



    def fit(self,X)       :
        nb_features=X.shape[1]
        self.centroids=X[0][0]*np.random.rand(self.nb_clusters,nb_features)
        affected_points=[[] for i in range(self.nb_clusters)]
        no_change=True
        iteration=0
        while no_change and iteration<self.max_iter:
            new_centroids=np.random.rand(self.nb_clusters,nb_features)
            for i in range(X.shape[0]):
                nearest_centroid=nearest(X[i],self.centroids,1)[0]
                affected_points[nearest_centroid].append(X[i])

            for k in range(self.centroids.shape[0]):
                
                points=np.array(affected_points[k])

                if points.shape[0]!=0:

                    new_centroids[k]=centre(points)

            if (np.linalg.norm(self.centroids-new_centroids)<self.eps):

                print("convergence")
                no_change=False
            
            print("old",self.centroids)
            print("new",new_centroids)
            self.centroids=new_centroids

            iteration+=1


    def predict(self,X):
        Y=np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            Y[i]=nearest(X[i],self.centroids,1)[0]
        return Y


    



            
                
                
                
                
            
            
                
                
                
        
        
        