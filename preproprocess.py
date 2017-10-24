# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 06:33:15 2017

@author:ishwar

WARNING!! don't use for final submission
copied parts from here http://learnandshare645.blogspot.fr/2016/06/feeding-your-own-data-set-into-cnn.html
"""  

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from numpy import *


##
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

##
img_x = 200
img_y = 200

#%%
# load images

imgpath = 'C:/Users/edmon_000/Desktop/mldm year 2/computer vision/VOCdevkit/VOC2007/JPEGImages/'
imgresizedpath = 'C:/Users/edmon_000/Desktop/mldm year 2/computer vision/VOCdevkit/VOC2007/input_data_pre_proprocessed/'

alloriginals = os.listdir(imgpath)
alloriginalssize = size(alloriginals)

for image in alloriginals:
    img = Image.open(imgpath + '/' + image)
    img = img.resize((img_x,img_y))
    img.save(imgresizedpath + '/' + image,'JPEG')
imlist = os.listdir(imgresizedpath)
imnum = len(imlist) #number of images

#size of 1 resized img
im1 = array(Image.open(imgresizedpath+'/'+imlist[0]))
m,n = im1.shape[0:2] 

#%%
# flatten the images

imgflattened = array([array(Image.open(imgresizedpath + '/' + ime)).flatten()
    for ime in imlist])

###TODO: for the labels search in the dictionary from the XML    

###TODO: training batch sets

###TODO: split into training and test set

###TODO: convert the class vectors into binary class matrices