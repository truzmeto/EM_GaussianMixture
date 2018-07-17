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
             
             k - number of nearest neighbours(k=1 by default)
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


def cross_val(data, labels, n_bags = 5, kmax = 15, seed = 1):
    '''
    Cross validate using elbow method to find optimum k.
    -----------
    
    Input:   data   - numpy ndarray of dimension (n_samples, n_features)
             kmax   - number of max k to try for cv
             labels - target variable of dim(n_samples)
             seed   - random seed, integer value
    Output:  k_opt  - opt value for k(scalar)
    '''
    size = len(data)
    acc_mat = np.empty(shape = [kmax, n_bags])

    for i in range(n_bags):

        resample_index = np.random.choice( size, size//2, replace = False)
        cv_set = data[resample_index,:]          
        cv_labs = labels[resample_index]
        accuracy = []

        for j in range(1,kmax+1):
            pred_labs = knn_main(train = data, test = cv_set, labels = labels, k = j)
            accuracy.append(getAccuracy(labels = cv_labs, pred_labels = pred_labs))

        #append each accur. array as column into np2D array    
        acc_mat[:,i] = np.array(accuracy)

        #get mean accura. over all bags
        mean_acc = np.mean(acc_mat, axis=1)

        #get index(k-val) that gives max accuracy
        opt_indx = np.argmax(mean_acc)
        k_opt = opt_indx + 1
    return k_opt, mean_acc 


#implement confu_mat calculation, accur, F1 also!    
def getAccuracy(labels, pred_labels):
    """
    Calc Acc given labels and pred_labels
    ---------------
    """
    
    correct = 0
    for i in range(len(labels)):
        if labels[i] == pred_labels[i]:
            correct += 1
    return (correct/float(len(labels))) * 100.0


def testing(train, test, labels, k_opt = 1):
    """
    Performs knn on testing set given opt k value.   
    
    -----------
    
    Input:   train & test - numpy ndarray of dimension (n_samples, n_features)
             k_opt        - optimum value for # of neirest neighbours    

    Output:  labels  - numpy ndarray of dimension (n_samples)
    """
    
    pred_labels = knn_main(train = train, test = test, labels = labels, k = k_opt)
    return pred_labels 
    
def FinaleMsg(n = 15, m = 50):
          """ Nicly decorated finale msg printing
          ---------------
          """
          pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
          return '\n'.join(pattern + ['WELL DONE'.center(m, '-')] + pattern[::-1])
