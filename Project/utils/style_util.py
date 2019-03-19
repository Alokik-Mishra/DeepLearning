import numpy as np
import tensorflow as tf
import scipy.io
import scipy.misc
from utils.vgg_util import vgg_func, layers_and_mean

#calculating content loss and style loss 

def content_loss_calculate(s, vgg):
    content_layer = 'conv4_2'
    a = s.run(vgg[content_layer])
    b = vgg[content_layer]
    shape = a.shape
    x = tf.reduce_sum(tf.pow(b - a, 2))

    y = 1 / (4 * shape[3] * (shape[1]*shape[2]))
   
    loss = y*x
    return loss

# calculating gram matrix for style 
def gram_matrix(x):
    gx = tf.matmul(tf.transpose(x),x)
    return gx
# calculate the style loss
# We are using 5 layers of the VGG model for the style loss. However, for the content loss, we are using just the output of one layer
def style_loss_calculate(s, vgg):
    
    style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
    # 
    wgt = [1.5,1.5,1.5,1.5,1.5]
    loss = 0
    n = len(wgt)
    for layer,i in zip(style_layers, range(n)):
        a = s.run(vgg[layer])
        b = vgg[layer]
        
        shape = a.shape
        
        # we are combining height and width to calculate the gram matrix. shape[3] represents number of filters
        
        g_a = gram_matrix(tf.reshape(a,(shape[1]*shape[2], shape[3])))   
        g_b = gram_matrix(tf.reshape(b,(shape[1]*shape[2], shape[3])))
        
        x = tf.reduce_sum(tf.pow(g_b - g_a, 2))
        y = 1 / (4 * (shape[3]**2) * (shape[1]*shape[2])**2)
        loss += wgt[i]*x*y
    return loss


 

