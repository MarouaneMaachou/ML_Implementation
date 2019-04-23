#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:13:44 2019

@author: maachou
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Kmeans import Kmeans
X,Y = make_blobs(cluster_std=0.7,random_state=20,n_samples=1000,centers=5)

Km=Kmeans(5)
Km.fit(X)
y_pred=Km.predict(X).reshape((-1,))

plt.scatter(X[:,0],X[:,1],c=y_pred)