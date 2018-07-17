## My implementation of some ML algorithms

### This repository contains some of ML algorithms implemented with powerful numpy library under Python3.

Currently it only has:

1. Expectation Maximization with Gaussian Mixture that performs clustering on data set with unknown labels.
   It only supports diagonal covariance meaniang std length is same along different feature directions. 

2. K-neirest neighbour classification algorithm including cross validation with bagging. Bagging is
   performed by resampling 50% with replacement, where n_bags can be passed as argument.  


