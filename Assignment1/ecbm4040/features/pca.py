import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # Implement PCA by extracting eigenvector     #
    ###############################################
    #Getting Cov matrix
    M, N = X.shape
    Cov_X = np.cov(X, rowvar=False)
    
    #Getting eigen decomp
    Eigenvals, Eigenvecs = np.linalg.eigh(Cov_X)
    
    #Sorting by eigenvalue
    Order = np.argsort(Eigenvals)[::-1]
    Eigenvecs = Eigenvecs[:,Order]
    Eigenvals = Eigenvals[Order]
    
    # Picking top K eigenvalue/eigenvector pairs
    Eigenvecs = Eigenvecs[:, :K]
    Eigenvals = Eigenvals[:K]
    
    # Computing output
    P = Eigenvecs.T
    T = Eigenvals
    
    
    return (P, T)
