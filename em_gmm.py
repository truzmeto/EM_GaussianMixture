import numpy as np
from scipy.spatial.distance import cdist

cents = np.array([[1,1],[1,10]])

def Estep(x_inp, x_cent):
    '''Performs the E step given input np.matrix and centroid locations as matrix '''
    #calculate distances from each centroid to all data points
    dist = cdist(x_inp, x_cent, 'euclidean')
    expect = np.empty(shape=[len(x_cent), len(x_inp)]).transpose()
    for i in range(len(x_cent)):
        #compute the standard deviations
        std = dist[:,i].std() 
        # Expectation step
        expect[:,i] = np.exp((-0.50 * dist[:,i]**2 / std))
        expect[:,i] = expect[:,i] / expect[:,i].sum()
    return expect


def Mstep(x_inp, expect):
    '''Performs the M step given input np.matrix and expectation values '''
    #Maximization Step
    centroids = np.dot(x_inp.transpose(),expect)
    return centroids.T


def simulate_EM(x_inp, x_cents, steps):
    for i in range(steps):
        expect = Estep(data, x_cents)
        x_cents = Mstep(data, expect)
    return x_cents

