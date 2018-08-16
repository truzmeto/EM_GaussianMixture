import numpy as np

def outlier_Zscore(X, thresh = 3.0):
    """
    Z-score method of outlier detection. After centring
    and rescaling the data, anything that is greater than
    threshold ( > |3|) should be considered an outlier and
    replaced with nan's for further processing
    --------------------------

    Input:   X    - normalized((x-x_mean)/x_std) numpy
                       ndarray of dim.(n_samps, n_feats)
             thresh  - threshhold value for detecting outliers

    Output:  out     - numpy ndarray of dim. (n_samps, n_feats)
    """
    col_mean = np.mean(X, axis = 0)
    col_std = np.std(X, axis = 0)
    data_Zscores = (X - col_mean) / col_std 
    X[np.abs(data_Zscores) > thresh] = np.nan

    return X


def outlier_mod_Zscore(X, thresh = 3.5):
    """
    Modified Z-score method for outlier detection. Since Z-score method
    relies on normal dist. assumption it doesn't work well with small data.
    Thus, modified Z-score method is used. It is calculated by deviding
    deviations from median by absolute deviations from median....
    --------------------------

    Input:   X       - numpy ndarray of dim.(n_samps, n_feats)
             thresh  - threshhold value for detecting outliers

    Output:  out     - numpy ndarray of dim (n_samps, n_feats)
    """
    
    col_median = np.median(X, axis = 0) # median per column (n)
    col_dev_median = X - col_median     # deviations from median (m,n)
    median_absolute_deviations = np.median(np.abs(col_dev_median), axis = 0) # (n)
    mod_Zscores = 0.6745 * col_dev_median / median_absolute_deviations # (m,n)
    X[np.abs(mod_Zscores) > thresh] = np.nan

    return X

