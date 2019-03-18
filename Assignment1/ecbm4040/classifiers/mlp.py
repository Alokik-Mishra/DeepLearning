from builtins import range
from builtins import object
import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

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
        num_layers = self.num_layers
        layers = self.layers
        ###################################################
        # Feedforward                                #
        ###################################################
        out_1 = layers[0].feedforward(X)
        for i in range(1, num_layers-1):
            out_1 = layers[i].feedforward(out_1)
        out_2 = layers[num_layers-1].feedforward(out_1)
        
        ###################################################
        # Backpropogation                            #
        ###################################################
        loss, dout = softmax_loss(out_2, y)
        
        ###################################################
        # Add L2 regularization                     #
        ###################################################
        square_weights = 0.0
        for i in range(num_layers):
            square_weights += np.sum(layers[i].params[0]**2)
                
        loss += 0.5*self.reg*square_weights
        
        dX  = layers[num_layers-1].backward(dout)
        self.layers[num_layers-1].gradients[0],self.layers[num_layers-1].gradients[1] = self.layers[num_layers-1].gradients
        for i in range(num_layers-2, -1, -1):
            dX = self.layers[i].backward(dX)
            self.layers[i].gradients[0],self.layers[i].gradients[1] = self.layers[i].gradients
        
        return loss

    def step(self, learning_rate=1e-5):
        num_layers = self.num_layers
        layers = self.layers
        reg = self.reg
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ###################################################
        # Use SGD  to update variables in layers     #
        ###################################################
        params = self.layers[0].params
        for i in range(1, num_layers):
            params = params + self.layers[i].params
        grads = self.layers[0].gradients
        for i in range(1, num_layers):
            grads = grads + self.layers[i].gradients
        

        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        
        for i in range(len(params)-1) :
            params[i] = params[i] - learning_rate*grads[i]

   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers

        int_score = np.dot(X, layers[0].params[0]) + layers[0].params[1]
        int_score[int_score<=0] = 0 
        
        for i in range(1, num_layers-1):
            int_score = np.dot(int_score, layers[i].params[0]) + layers[0].params[1]
        
        scores = np.dot(int_score, layers[num_layers-1].params[0]) + layers[num_layers-1].params[1]
        predictions = np.argmax(scores, axis = 1)
             
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
        
        


