import pandas as pd
import numpy as np

def add_bias(X):
    """
    Function to add bias unit(column composed of ones) to a np.ndarray.
     ---------------
    
    input:   X   - input data, np.ndraay of dim(n_samples, n_features)     -  
    
    output:  X   - output data, np.ndarray of dim(n_samples, n_features+1)  
    """
    
    n_samples = X.shape[0]
    bias = np.ones((n_samples, 1))
    return np.concatenate((bias, X), axis=1)


def sigmoid(z):
    """ Sigmoid function"""
    return 1 / (1 + np.exp(-z))

def cost(h, y):
    """
    Cost function or Log Loss function is subject
    to minimization to find optimal weights
    """   
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()    


def getAccuracy(labels, pred_labels, method = "class"):
    """
    Calc Acc given labels and pred_labels
    when method = "class"  - it returns # of correctly predicted labels, normalized to 1 
    when method = "reg"    - it returns 1 - sum((y-y_hat)^2) / sum((y-y_mean)^2)
    ---------------
    
    input:   labels       - actual known labels(values in case of regression)
             pred_labels  - predicted labels(ues in case of regression)
             method       - can be either classification(class-default) or regression(reg) 
    
    output:  accuracy % 
    """

    n = len(labels)
    #-------- check point
    if n != len(pred_labels):
        print("Two vectors must have equal size!")
      
    
    correct = 0.0
    if method == "class":  #just count correct cases 
        for i in range(n):
            if labels[i] == pred_labels[i]:
                correct += 1
        correct = correct/float(n) 
        
    if method == "reg": #return root mean squared error
        ave = np.mean(labels)
        norm = 0.0
        correct = 0.0
        for i in range(n):
            correct = correct + (labels[i] - pred_labels[i])*(labels[i] - pred_labels[i])
            norm = norm + (labels[i] - ave)*(labels[i] - ave)
            
        acc = 1.0 - np.sqrt(correct / norm)    
        correct = acc
        
    return correct



def initilize_weights(n_features):
    """
    Function to initilize weights(slope parameters).
     ---------------
    
    input:   n_features   - number of features in data including bias unit   
    
    output:  theta        - initial theta values np.ndarray of size (n_features)  
    """
    #theta = np.zeros(n_features)
    theta = np.random.random(n_features)
    return theta      


def train(X, y, num_iters=2000, lr=0.02, Lambda = 0.1):
    """
    This function finds optimal values for theta by performing gradient descent.
     ---------------
    
    input:   X          - input data, np.ndraay of dim(n_samples, n_features) 
             y          - actual known labels of input data
             num_iters  - number of iterations for grad descent calculation    
             lr         - learning rate
             Lambda     - regularization parameter(Ridge  - l2 penalty)
             
    output:  theta      - optimal theta values calculated, np.ndarray of size (n_features)  
    """
    #weights initialization
    n_features = X.shape[1]
    n_samples = X.shape[0]
    theta = initilize_weights(n_features)

    for i in range(num_iters):
        z = np.dot(X, theta)
        h = sigmoid(z)
        
        #calculate gradient
        gradient = np.dot(X.T, (h - y)) / n_samples  + Lambda*theta / n_samples
        #update weights
        theta -= lr * gradient

    return theta       



#def train_sgd(X, y, num_iters=2000, lr=0.02):
#    """
#    This function finds optimal values for theta by performing
#    stochastic gradient descent, where instead of updating all
#    weight we randomly choose one and update it.
#     ---------------
#    """
        #rendomly select target weight
#       sel_id = np.random.randint(0,n_features)
        #update selected theta
#       theta[sel_id] -= lr * gradient[sel_id]
#    return theta


def train_sgd(X, y, batch_frac = 0.2 , num_iters=1000, lr=0.02, Lambda = 0.1):
    """
    This function finds optimal values for theta by performing stochastic gradient descent.
    It updates wights using random subset of data(mini batch) for each iteration(epoch).
    ----------------
    
    input:   X          - input data, np.ndraay of dim(n_samples, n_features) 
             y          - actual known labels of input data
             num_iters  - number of iterations(epochs) for grad descent calculation    
             lr         - learning rate
             batch_frac - fraction of data to determine mini batch size
             Lambda     - regularization parameter(Ridge  - l2 penalty)

    output:  theta      - optimal theta values calculated, np.ndarray of size (n_features)  
    """
    #weights initialization
    n_features = X.shape[1]
    n_samples = X.shape[0]
    theta = initilize_weights(n_features)
    batch_size = int(n_samples * batch_frac)
    
    for i in range(num_iters):
           
        resample_index = np.random.choice(n_samples, batch_size, replace = False)
        batchX = X[resample_index]   # data[resample_index,:]          
        batchY = y[resample_index]

        z = np.dot(batchX, theta)
        h = sigmoid(z)
        
        #calculate gradient
        gradient = np.dot(batchX.T, (h - batchY)) / batch_size + Lambda*theta / batch_size
        #update weights
    
    return theta       



def test(X,theta):
    """
    This function uses optimal theta values to make prediction
    on any data set(training or testing).
    ---------------
    
    input:   X          - input data, np.ndraay of dim(n_samples, n_features) 
             theta      - optimal theta values calculated, np.ndarray of size (n_features)  

    output:  pred       - predicted class, ither 0 or 1, np.ndarray of size (n_samples)  
    """
    pred = sigmoid(np.dot(X, theta))
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    
    return pred

