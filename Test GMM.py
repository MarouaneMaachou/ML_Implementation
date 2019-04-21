#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 02:43:24 2019

@author: maachou
"""

from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from GMM import GMM        
mix=GMM(K=6)   
X,Y = make_blobs(cluster_std=0.5,random_state=20,n_samples=100,centers=6)
plt.scatter(X[:,0],X[:,1])
print(X.shape)
mix.fit(X)
mix.Means()
Y=mix.predict(X)
plt.scatter(X[:,0],X[:,1],c=Y)