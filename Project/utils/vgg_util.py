import numpy as np
import tensorflow as tf
import scipy.io
import scipy.misc

# source of vgg-19 : http://www.vlfeat.org/matconvnet/pretrained/
vgg19 = scipy.io.loadmat("imagenet-vgg-verydeep-19.mat")

def layers_and_mean():
    layers = vgg19['layers']
    mean = vgg19['meta']['normalization'][0][0][0][0][2][0][0]
    
    return layers, mean

#Defining layers of the connvolution neural network

#loading weights and bias of the trained model
# Defining Convolution layers to use the weight and biases of the trained model

def conv_relu(layer0, layer1):
    layers,_ = layers_and_mean()
    weight = layers[0][layer1][0][0][2][0][0]
    bias = layers[0][layer1][0][0][2][0][1]
    w = tf.constant(weight)
    b = tf.constant(np.reshape(bias, bias.size))
    con_layer = tf.nn.conv2d(layer0, filter = w, strides=[1,1,1,1], padding='SAME') + b
    # relu activation function
    relu_layer = tf.nn.relu(con_layer)
    return relu_layer
    
#pooling layer, average or max based on the type of pooling
def pooling(input, type):
    if type == "max_pooling":
         return tf.nn.max_pool(input, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
    else:
        return tf.nn.avg_pool(input, ksize=(1,2,2,1), strides=(1,2,2,1),padding='SAME')
    
 # Defining layers to operate the content and style image in different layers. In these layesrs we are calling conv_relu
# which calculates the output based on the weights of the trained model

def vgg_func(height, width):
    vgg = {}
    pool = "average_pooling"

    vgg['input']= tf.Variable(np.zeros((1, height, width, 3)), dtype = 'float32')

    vgg['conv1_1'] = conv_relu(vgg['input'], 0)        #layer 0 and 1(relu)
    vgg['conv1_2'] = conv_relu(vgg['conv1_1'], 2)      #layer2 and 3
    vgg['pool1'] = pooling(vgg['conv1_2'], pool)         #layer 4

    vgg['conv2_1'] = conv_relu(vgg['pool1'], 5)        #layer 5 and 6
    vgg['conv2_2'] = conv_relu(vgg['conv2_1'], 7)      #layer 7 and 8
    vgg['pool2'] = pooling(vgg['conv2_2'], pool)         #layer 9

    vgg['conv3_1'] = conv_relu(vgg['pool2'],10)        #layer 10 and 11 
    vgg['conv3_2'] = conv_relu(vgg['conv3_1'], 12)     #layer 12 and 13
    vgg['conv3_3'] = conv_relu(vgg['conv3_2'], 14)     #layer 14 and 15
    vgg['conv3_4'] = conv_relu(vgg['conv3_3'], 16)     #layer 16 and 17
    vgg['pool3'] = pooling(vgg['conv3_4'], pool)         #layer 18   

    vgg['conv4_1'] = conv_relu(vgg['pool3'],19)        #layer 19 and 20
    vgg['conv4_2'] = conv_relu(vgg['conv4_1'], 21)     #layer 21 and 22
    vgg['conv4_3'] = conv_relu(vgg['conv4_2'], 23)     #layer 23 and 24
    vgg['conv4_4'] = conv_relu(vgg['conv4_3'], 25)     #layer 25 and 26 
    vgg['pool4'] = pooling(vgg['conv4_4'], pool)         #layer 27

    vgg['conv5_1'] = conv_relu(vgg['pool4'],28)        #layer 28 and 29
    vgg['conv5_2'] = conv_relu(vgg['conv5_1'], 30)     #layer 30 and 31
    vgg['conv5_3'] = conv_relu(vgg['conv5_2'], 32)     #layer 32 and 33
    vgg['conv5_4'] = conv_relu(vgg['conv5_3'], 34)     #layer 34 and 35
    vgg['pool5'] = pooling(vgg['conv5_4'], pool)         #layer 36
    
    return vgg


