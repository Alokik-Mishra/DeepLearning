from __future__ import print_function

import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:
    input -> DenseLayer -> AffineLayer -> softmax loss -> output
    Or more detailed,
    input -> affine transform -> ReLU -> affine transform -> softmax -> output

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_dim=3072, hidden_dim=200, num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.layer1 = DenseLayer(input_dim, hidden_dim, weight_scale=weight_scale)
        self.layer2 = AffineLayer(hidden_dim, num_classes, weight_scale=weight_scale)
        self.reg = reg

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        N = X.shape[0]
        b1 = self.layer1.params[1]
        W1 = self.layer1.params[0]
        
        b2 = self.layer2.params[1]
        W2 = self.layer2.params[0] 
        ###################################################
        # Feedforward                                #
        ###################################################
        out_1 = self.layer1.feedforward(X)
        out_2 = self.layer2.feedforward(out_1)
        ###################################################
        # Backpropogation, here is just one dense         #
        # layer, it should be pretty easy                 #
        ###################################################
        loss, dout = softmax_loss(out_2, y)
        
        # Add L2 regularization
        square_weights = np.sum(self.layer1.params[0]**2) + np.sum(self.layer2.params[0]**2)
        loss += 0.5*self.reg*square_weights
        

        dX1  = self.layer2.backward(dout)
        self.layer2.gradients[0],self.layer2.gradients[1] = self.layer2.gradients
        dX_2 = self.layer1.backward(dX1)
        self.layer1.gradients[0],self.layer1.gradients[1] = self.layer1.gradients
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        # creates new lists with all parameters and gradients
        layer1, layer2 = self.layer1, self.layer2
        params = layer1.params + layer2.params
        grads = layer1.gradients + layer2.gradients
        
        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        ###################################################
        # Use SGD or SGD with momentum to update          #
        # variables in layer1 and layer2                  #
        ###################################################
        #if optim == 'SGD':
        for i in range(len(params)) :
            params[i] = params[i] - learning_rate*grads[i]
        #if optim == 'momentum':
         #   v = momentum * learning_decay + learning_rate*grads
         #   params = params - v
   
        # update parameters in layers
        layer1.update_layer(params[0:2])
        layer2.update_layer(params[2:4])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        layer1, layer2 = self.layer1, self.layer2
        #######################################################
        # Remember to use functions in class SoftmaxLayer     #
        #######################################################
        layer_1 = np.dot(X, self.layer1.params[0]) + self.layer1.params[1]
        layer_1[layer_1<=0] = 0 # ReLu
        scores = np.dot(layer_1, self.layer2.params[0]) + self.layer2.params[1]
        predictions = np.argmax(scores, axis=1)
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
    
    def save_model(self):
        """
        Save model's parameters, including two layer's W and b and reg
        """
        return [self.layer1.params, self.layer2.params, self.reg]
    
    def update_model(self, new_params):
        """
        Update layers and reg with new parameters
        """
        layer1_params, layer2_params, reg = new_params
        
        self.layer1.update_layer(layer1_params)
        self.layer2.update_layer(layer2_params)
        self.reg = reg
        
        
        


