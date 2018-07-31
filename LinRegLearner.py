"""
A simple wrapper for linear regression.  (c) 2018 T. Ruzmetov
"""

import numpy as np

def train(training, dataY):
    """
    #slap on 1s column so linear regression finds a constant term
    newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
    newdataX[:,0:dataX.shape[1]]=dataX
    """
    # build and save the model
    #return model_coefs, residuals, rank, s = np.linalg.lstsq(training, dataY)
    return  np.linalg.lstsq(training, dataY)
        
def query(testing):
    """
    
    """
    return (model_coefs[:-1] * testing).sum(axis = 1) + model_coefs[-1]
