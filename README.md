# Deep Learning
### Fall 2018 - Electrical Enginerring, Columbia Univesity
### Prof. Zoran Kostic

## Overview
1. [Course description](#desc)
2. [Tech/framework](#tech)
3. [Assignment 1](#as1)
4. [Assignment 2](#as2)
5. [Final Project](#finpro)

<a name="desc"></a>
## Course Description
ECBM4040 is a 1 semester course covering deep learning theory and implementation both from scratch and using tensorflow. The course covers the following topics:

* Machine Learning, representation learning, and neural nets
* Network architecure, backprop, and loss functions
* Optimization and regularization
* Multilayer perceptrons (MLP), feedforward networks, and fully-connected nets
* Convolutional neural networks (CNN) and image classification
* Recurrent neural networks (RNN) and LSTMs
* Autoencoders
* Generative adversarial networks (GANs)

<a name="tech"></a>
## Tech/Frameworks
Most of the analysis is done using Jupyter Notebooks in python 3. Util files are .py. Most of the code uses basic libraries including numpy, and matplotlib.
Some of the models are made using the tensorflow API and displayed using tensorboard.

<a name="as1"></a>
## Assignment 1
Assignment one consists of three tasks, starting with building and comparing the performance of the linear SVM classfier and the softmax classfier. Next, I build a multilayer perceptron with stochastic gradient descent from scratch and then using the tensorflow API. Finally, I use dimensionality reduction using a PCA and see how that affects the performace of the MLP, as well as how using a 2 layer net affects the tSNE visualizations.

I work on the CiFAR-10 dataset of images with corresponding labels; see below for examples.

<img src="/Assignment1/images/ci_FAR_ex.png" width="354">

In the first task, I build linear SVM and softmax classifiers using sdg optimization. The two charts below show the performance of the models.

<img src="/Assignment1/images/LinSVM.png" width="354">  <img src="/Assignment1/images/Softmax.png" width="354">

In both cases I use a regularization parameter of 1e-5 and a learning rate of 1e-7. While the charts show that the loss is first order lower in the Softmax, after 1500 iterations, the validation accuracy of the SVM at 0.3 is higher than that of the softmax classfier at 0.23. The runtime for both models with 1500 iterations is negligible.

I then build a multilayer perceptron using the same data from scratch and check my code using a simple version built in tensorflow. Using a two layer network, I get the following classification performance.

<img src="/Assignment1/images/twolayernetwork.png" width="354">

The network architecture is as follows

```python
model = TwoLayerNet(input_dim=3072, hidden_dim=100, num_classes=10, reg=1e-4, weight_scale=1e-3)
num_epoch = 10
batch_size = 500
lr = 5e-4
```
This achieves a validation set classification accuracy of 0.449 and a test set accuracy of 0.436.

I then echance the model by increasing the width of the hidden layer in an attempt to get at least 50% testing set accuracy. 

```python
model = TwoLayerNet(input_dim=3072, hidden_dim=200, num_classes=10, reg=1e-4, weight_scale=1e-3)
num_epoch = 10
batch_size = 100
lr = 5e-4
```
The model performance can be seen below.

<img src="/Assignment1/images/twolayernetwork_enchanced.png" width="354">

The model reaches a test set accuracy of 0.51 which meets the 50% benchmark I wanted.
Next I change the architecture by decreasing the width of the hidden layer, but adding a layer and adjusting some other hyperparameters:

```python
model = MLP(input_dim=3072, hidden_dims=[100, 100], num_classes=10, reg=0.1, weight_scale=1e-3)
num_epoch = 10
batch_size = 500
lr = 1e-2
```

While the new model does better on the training set classfication, it only achieves a testing set accuracy of 0.45. I conclude the section by replicating the net architecture in tensorflow.

In the last part of the assignment I use PCA to look at whether dimensionality reduction help with the performance of the MLP. 

<img src="/Assignment1/images/PCA_orig.png" width="250"> <img src="/Assignment1/images/PCA.png" width="250">  <img src="/Assignment1/images/PCA_loadings.png" width="250"> 

The images above show the orginal images (left), the principal components (center), and the recompiled image with the factor loadings (right). 

```python
model = MLP(input_dim=1500, hidden_dims=[100, 100], num_classes=10, reg=0.1, weight_scale=1e-3)
num_epoch = 10
batch_size = 500
lr = 1e-2
```
The model with the above architecture on the reduced dimension data underperforms with a validation set accuracy of only 0.28.

Finally, I compare the well known tSNE vislizations of the raw data with the data after running it through a two layer network. The two versions can be compared below, with the raw data version (left) being much more noisy than the one after the two layer network (right). Implemening a 100 iterations for both involves a runtime of 23 secs for the raw data and 24 secs for the data after running it through the model.

<img src="/Assignment1/images/tSNE1.png" width="354"> <img src="/Assignment1/images/tSNE2.png" width="354"> 

<a name="as2"></a>
## Assignment 2
In this assignment, I start out by comparing different optimization techniques such as sgd, rmsprop, adam, etc. and then move on to experiment with two different regularization techniques, dropout and batch normalization. Next I build a convolutional neural net, both from scratch and with tensorflow. Finally I look at data augmentation techniques often used in image recongition.

The optimization techniques I compare are SGD, SGD with momentum, RMSprop, Adam, and Nadam. I make sure to use the same MLP model for all four with the same hyperparameters:

```python
model = MLP(input_dim=3072, hidden_dims=[100, 100], num_classes=10, l2_reg=0.0, weight_scale=1e-3)
```

The results can be seen below:

<img src="/Assignment2/images/opt_training.png" width="250"> <img src="/Assignment2/images/opt_accuracy.png" width="250"> <img src="/Assignment2/images/opt_validation_accuracy.png" width="250"> 

In all three measures, but most importantly the validation accuracy, the RMSprop alogrithm to optimize the gradient seems to work best. It should be noted that this is not a foolproof assesment, there are many pros and cons to each of the algorithms used, and while RMSprop worked best here, it is not be the universally preferred algorithm.

I next look at dropout and batch normalization which are two regularization techniques that help to minimize generalization error. 

I compare various 'dropout rates' while keeping the MLP model and its hyperparameters constant, as well as sticking with the Adam optimizer algorithm.

```python
model = MLP(input_dim=3072, hidden_dims=[200], num_classes=10, 
            weight_scale=1e-3, l2_reg=0.0, dropout_config=dropout_config)
optimizer = AdamOptim(model)
```

The results of various different dropout rates can be seen below:

<img src="/Assignment2/images/dropout_loss.png" width="250"> <img src="/Assignment2/images/dropout_accuracy.png" width="250"> <img src="/Assignment2/images/dropout_accuracy_validation.png" width="250"> 

Certain pattens emerge when looking at the validation set accruacy. Initially the low dropout rates underperform, however they do eventually catch up to the other higher rates, and ultimately the best performing model is the 0.1 rate. with a validation accuracy of 0.358 (although this is marginal, the next highest a rate of 0.5 with an accuracy of 0.352).

The other regularization technique I examine is batch normalization. I implement batch normalization on both a shallow and a deep mlp and find that the beneficial effects in terms of validation accuracy are much higher on deeper architectures. 

The next part of the assignment involves building a CNN. I first build the model from scratch using only numpy to build convolution and max pool layers. Then i code a model using tensoflow. 
