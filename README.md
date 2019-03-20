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
ECBM4040 is a 1 semester course covering deep learning theory and implementation from both basic libraries and tensorflow. The course covers the following topics:

* Machine Learning, representation learning, and neural nets
* Network architecure, backprop, and loss functions
* Optimization and regularization
* Multilayer perceptrons (MLP), feedforward networks, and fully-connected nets
* Convolutional neural networks (CNN) and image classification
* Recurrent neural networks (RNN) and LSTMs
* Autoencoders
* Generative adversarial networks (GANS)

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



