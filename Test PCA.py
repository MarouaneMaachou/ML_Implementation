#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 03:22:03 2019

@author: maachou
"""

from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from PCA import PCA
pca=PCA(output_dim=2)
data=load_boston()
X=data.data
print(X.shape)
print(pca.fit_transform(X).shape)