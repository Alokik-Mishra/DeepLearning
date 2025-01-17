import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Compute the softmax loss and its gradient using explicit loops.           #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        score = X[i].dot(W)
        score -= np.max(score)
        
        den_sum = 0.0
        for j in score:
            den_sum += np.exp(j)
        loss -= score[y[i]]
        loss += np.log(den_sum)
        
        for k in range(num_classes):
            C = 1.0 / den_sum * np.exp(score[k])
            dW[:,k] += C * X[i]
            if k == y[i] :
                dW[:, k] -= X[i]
    
    ## Averaging and Regularization  
    loss /= num_train
    dW /= num_train
      
    loss += 0.5 * reg * np.sum(W*W)
    dW += reg*2*W          
 
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Compute the softmax loss and its gradient using no explicit loops.        #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    score = X.dot(W)
    score -= np.matrix(np.max(score, axis=1)).T

    num = score[np.arange(num_train), y]
    den_sum = np.sum(np.exp(score), axis=1)
    den = np.log(den_sum)
    loss = np.sum(- num + den)

    
    d = np.exp(score) / np.matrix(den_sum).transpose()
    d[np.arange(num_train),y] -= 1
    dW = X.transpose().dot(d)
    ## Averaging and regularization
    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W
    dW = np.asarray(dW)

    return loss, dW
