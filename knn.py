"""
Talant Ruzmetov, 2018
A simple wrapper for KNN implementation for classification and regression. 
"""

import numpy as np
from scipy.spatial.distance import cdist

def knn_main(train, test, labels, k = 1):
    
    '''
    Main part of knn, where return predicted labels based on
    proximity of instnace to other near points.
    -----------
    
    Input:   train - numpy ndarray of dimension (n_train_samples, n_features)
             test  - numpy ndarray of dimension (n_test_samples, n_features)
             
             k - number of nearest neighbours
             labels - given trainig data labels, np array (size = n_train_labels)
   
    Output:  pred_labels - np array(size = n_train_labels  or n_test_labels) 
    '''

    dist = cdist(train, test, 'euclidean')
    pred_labels = []

    #loop over columns of dist matrix
    n = dist.shape[1]
    for i in range(n):

        #sort dist vals in increasing order and return indecies
        indx = np.argsort(dist[:,i])

        #drop 0 element(self dist) and return up to k-neighbors
        indx_save = indx[1:k+1]

        #retrive labels for k-neighbors
        k_labels = labels[indx_save]       
        pred_labels.append(np.bincount(k_labels).argmax())

    return np.array(pred_labels)     


def cross_val(data, labels, p = 0.4, kmax = 15):
    '''
    Cross validate using elbow method to find optimum k.
    
    -----------
    
    Input:   data   - numpy ndarray of dimension (n_samples, n_features)
             kmax   - number of max k to try for cv
             labels - target variable of dim(n_samples)
             seed   - random seed, integer value
    Output:  k_opt  - opt value for k(scalar)
    '''
    
    test = data[0:int(p*len(data))]        #cv chunk, sample randomly! 
    cv_labels = labels[0:int(p*len(data))] #labels for sampled chunk!
    
    accuracy = []
    for i in range(1,kmax):
        pred_labels = knn_main(train = data, test = test, labels = labels, k = i)
        accuracy.append(getAccuracy(labels = cv_labels, pred_labels=pred_labels))

    accuracy = np.array(accuracy)
    opt_indx = np.argmax(accuracy)
    k_opt = opt_indx
    return k_opt, accuracy 


#implement confu_mat calculation, accur, F1 also!    
def getAccuracy(labels, pred_labels):
    """ Calc Acc given labels and pred_labels """
    correct = 0
    for i in range(len(labels)):
        if labels[i] == pred_labels[i]:
            correct += 1
    return (correct/float(len(labels))) * 100.0


def test(train, test, k_opt = 1):
    '''
    Performs knn on testing set given opt k value.   
    
    -----------
    
    Input:   train & test - numpy ndarray of dimension (n_samples, n_features)
             k_opt        - optimum value for # of neirest neighbours    

    Output:  labels  - numpy ndarray of dimension (n_samples)
    '''
    
    pred_labels = knn_main(train = train, test = test, labels = labels, k = k_opt)
    accuracy.append(getAccuracy(labels, pred_labels))
    accuracy = np.array(accuracy)

    return accuracy 
    

