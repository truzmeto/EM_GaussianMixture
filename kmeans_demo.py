#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist


def Initilize_centroids(x_inp, n_centroids, seed):
    '''
    Initilize centroids by randomly selecting from data range.
    
    -----------
    
    Input:   x_inp - numpy ndarray of dimension (n_samples, n_features)
             n_centroind  - number of components, means or centroids.
             seed - random seed, integer value
    Output:  init_means - numpy ndarray of dimension (n_centroids, n_features)
    '''
    #set rand.seed
    np.random.seed(seed)
    x_max, x_min  = x_inp.max(axis=0), x_inp.min(axis=0)
    n_features = x_inp.shape[1]
    init_means = np.random.rand(n_centroids, n_features) * (x_max - x_min) + x_min
    return init_means
    
   
    

def UpdateLabels(x_inp, means, n_centroids):
    '''
    Calculates distances between the all data poins and centroids,
    then assignes label to each data point based on proximity to
    centroid position.
    -----------
    
    Input:   x_inp - numpy ndarray of dimension (n_samples, n_features)
             means - centroid positions, numpy ndarray of dimension (n_centroids, n_features)   
             n_centroind  - number of components( means or centroids )
    Output:  dist  - numpy ndarray of dimension (n_samples, n_centroids)
             labels - numpy ndarray of dimension(n_samples)   
    '''
    
    #do it via scipy
    dist = cdist(x_inp, means, 'euclidean')
    
    #or do it ur way
    #for i in range(len(means)):
    #    dist = x_inp - means[i]
    #    dist = dist*dist ??????????????
    
    #assign labels
    labels = dist.argmin(axis=1) # very elegant way of labeling!
   
    return dist, labels


def UpdateMeans(x_inp, labels, n_centroids):
    """
    Updates centroid positions by moving centroids to
    geometric center of mass for a given group,based on new labels.
    -------------------
    
    Input:   x_inp - numpy ndarray of dimension (n_samples, n_features)
             labels - numpy ndarray of dimension(n_samples)   
    
    Output:  centers - updated centroid positions
    """
    centers = []
    for i in range(n_centroids):
        centers.append(x_inp[labels == i].mean(axis=0))
        
    centers = np.array(centers)
    
    return centers

    

def TrainKmeans(x_inp, n_centroids, n_iters, seed):
    '''
      
  
    -----------
    
    Input:   x_inp - numpy ndarray of dimension (n_samples, n_features)
             n_centroind  - number of components( means or centroids )
    Output:  means  - numpy ndarray of dimension (n_centroids, n_features)
             labels - clustering labels based on finale expectation value comparison  
    '''
    
    
    means = Initilize_centroids(x_inp, n_centroids, seed)   
    save_change =  []
    
    for i in range(n_iters):
       
        dist, new_labels = UpdateLabels(x_inp, means, n_centroids)
        new_means = UpdateMeans(x_inp, new_labels, n_centroids)
        means = new_means.copy()
        
    
    return means, new_labels
