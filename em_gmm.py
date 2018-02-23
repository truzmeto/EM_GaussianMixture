"""
A simple wrapper for linear regression.  (c) 2018 Talant Ruzmetov
"""

import numpy as np
from scipy.spatial.distance import cdist

def Initilize_centroids(x_inp, n_centroids = 2, seed =100):
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



def Estep(x_inp, means, n_centroids):
    '''
    Performs Expectation step given input data and returns expectation values.   
    
    -----------
    
    Input:   x_inp - numpy ndarray of dimension (n_samples, n_features)
             means - centroid positions, numpy ndarray of dimension (n_centroids, n_features)   
             n_centroind  - number of components( means or centroids )
    Output:  x_expect  - numpy ndarray of dimension (n_samples, n_centroids)
    '''
    
    dist = cdist(x_inp, means, 'euclidean')
    x_expect = np.empty(shape=[len(x_inp), n_centroids])
    
    for i in range(n_centroids):
        #compute the standard deviations
        std = dist[:,i].std() 
        #Expectation step
        x_expect[:,i] = np.exp((-0.50 * dist[:,i]**2 / std))
        x_expect[:,i] = x_expect[:,i] / x_expect[:,i].sum()
    return x_expect


def Mstep(x_inp, x_expect):
    '''
    Performs Maximization step given input data and expectation values, then
    and returns centroid positions.   
    
    -----------
    
    Input:   x_inp - numpy ndarray of dimension (n_samples, n_features)
             x_expect  - numpy ndarray of dimension (n_samples, n_centroids)
    Output:  means  - numpy ndarray of dimension (n_centroids, n_features)   
    '''
    
    #Maximization Step
    means = np.dot(x_inp.transpose(),x_expect)
    return means.T


def train_EM(x_inp, n_centroids, n_steps):
    '''
    Here we simulate EM by iterating up to n_step and return finale centroid positions.   
    
    -----------
    
    Input:   x_inp - numpy ndarray of dimension (n_samples, n_features)
             n_centroind  - number of components( means or centroids )
    Output:  means  - numpy ndarray of dimension (n_centroids, n_features)
             labels - clustering labels based on finale expectation value comparison  
    '''
    
    seed = 101
    means = Initilize_centroids(x_inp, n_centroids, seed)   
        
    for i in range(steps):
        x_expect = Estep(data, means, n_centroids)
        means = Mstep(data, x_expect)
        
    labels = x_expect.argmax(axis=1) # very elegant way of labeling!
    return means, labels



















