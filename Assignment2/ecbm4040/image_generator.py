#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate 


class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        self.x = x.copy()
        self.y = y.copy()
        self.N = x.shape[0]
        self.height = x.shape[1]
        self.weight = x.shape[2]
        self.num_translated = 0
        self.rot = 0
        self.is_horizontal_flip = False
        self.is_verticle_flip = False
        self.is_add_noise = False


        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.N_aug = self.N

    def create_aug_data(self):       
        '''
        Combine all the data to form a augmented dataset 
        '''
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """


        N = self.N
        y = self.y
        X = self.x
        tot_batches = N // batch_size
        batch_count = 0
        while True:
            if batch_count < tot_batches:
                batch_count += 1
                batch_start = batch_count*batch_size
                batch_end = (batch_count + 1)* batch_size
                yield((X[batch_start:batch_end,:,:,:], y[batch_start:batch_end,]))
            else :
                if shuffle == True:
                    np.shuffle(X)
                    batch_count = 0
                else:
                    batch_count = 0
            
        

    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        r = 4
        f, axarr = plt.subplots(r, r, figsize=(8,8))
        for i in range(r):
            for j in range(r):
                img = images[r*i+j]
                axarr[i][j].imshow(img, cmap="gray")

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """

        X = self.x
        y = self.y
        X_int = np.roll(X, shift_height, axis = 1)
        X_int = np.roll(X_int, shift_width, axis = 2)
        num_translated = X.shape[1] * shift_height + X.shape[2] * shift_width
        
        self.num_translated = num_translated
        self.is_translated = True
        self.translated = (X_int, y)
        
        return(X_int, y)

    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
        """

        X = self.x
        y = self.y
        X_int = rotate(input = X, angle = angle, axes = (1,2))
        
        self.rot = angle
        self.rotated = (X_int, y)
        return(X_int, y)
        

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        X = self.x
        y = self.y
        if mode == 'h':
            X_int = np.flip(X, axis = (1))
            self.is_horizontal_flip = True
        elif mode == 'v':
            X_int = np.flip(X, axis = (2))
            self.is_verticle_flip = True
        elif mode == 'hv' :
            X_int = np.flip(X, axis = (1,2))
            self.is_verticle_flip = True
            self.is_horizontal_flip = True
            
        self.flipped = (X_int, y)
        return(X_int, y)
        
        

    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """

        X = self.x
        y = self.y
        
        portion = int(X.shape[0] * portion)
        mask = np.random.choice(a = X.shape[0], size = portion , replace=False)
        Noise = np.zeros_like(X)
        Noise[mask,:, :, :] = np.random.normal(amplitude, 20)
        X_int = X + Noise   
        self.is_add_noise = True
        self.added = (X_int, y)
        
        return(X_int, y)
