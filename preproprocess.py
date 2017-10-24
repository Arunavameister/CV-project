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



from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


import glob
import xml.etree.ElementTree
import re

#%% definitions


#### change path according to your computer

#this is the path to the XML directory of all the images
xmlpath = "C:/Users/edmon_000/Desktop/mldm year 2/computer vision/VOCdevkit/VOC2007/Annotations/"
imgpath = 'C:/Users/edmon_000/Desktop/mldm year 2/computer vision/VOCdevkit/VOC2007/JPEGImages/'
imgresizedpath = 'C:/Users/edmon_000/Desktop/mldm year 2/computer vision/VOCdevkit/VOC2007/input_data_pre_proprocessed/'

allclassfiles = glob.glob(xmlpath+'*.xml')

img_x = 200
img_y = 200

#dictionary of num and class
classes ={0:'aeroplane',
        1:'bicycle',
        2:'bird',
        3:'boat',
        4:'bottle',
        5:'bus',
        6:'car',
        7:'cat',
        8:'chair',
        9:'cow',
        10:'diningtable',
        11:'dog',
        12:'horse',
        13:'motorbike',
        14:'person',
        15:'pottedplant',
        16:'sheep',
        17:'sofa',
        18:'train',
        19:'tvmonitor'}


#i forgot that i needed numbers from text. better to invert it now
classes = {obj: num for num, obj in classes.items()}

#these will be used to create the dictionary images_dict which maps image id with numerical class (0..19). 
xml_imgnums    = []
xml_imgclasses = []
# read all xml and load into a tuple
# e[1] contains the image number
# e[6][0] contains image class
for f in allclassfiles:
    e = xml.etree.ElementTree.parse(f).getroot()
    imgnum = e[1].text
    xml_imgnums.append(int(re.sub('\.jpg$','',imgnum)))
    xml_imgclasses.append(classes.get(e[6][0].text))
    #print(imgnum + " " + e[6][0].text + " " + str(classes.get(e[6][0].text)))


# we now create a dictionary of the image number and its respective class
images_dict = dict(zip(xml_imgnums, xml_imgclasses))
#%%
# load jpg images

alloriginals = [f for f in os.listdir(imgpath) if os.path.splitext(f)[-1] == '.jpg']
alloriginalssize = size(alloriginals)

final_imgs = []
final_imgclasses = []

# resizes and stores images if the jpg images are found in the dictionary images_dict above
for image in alloriginals:
    #now check if the jpg file we are reading exists in the images_dict, if so  lets create new lists   
    # :-4 to remove the .jpg 
    imagenum = int(image[:-4])
    if images_dict[imagenum]:
        
        #first add the key to new list
        #final_imgnums.append(imagenum)
        
        #then add it's corresponding class retrieved from the xml dictionary
        final_imgclasses.append(images_dict[imagenum])
       
        #now save the good resized images (ones which are in the dictionary)
        im = Image.open(imgpath + '/' + image)
        img = im.resize((img_y,img_x))
        
        # this finally becomes a matrix of the dimension (number of images) * (img_x * img_y)
        final_imgs.append(img.flatten())
        img.save(imgresizedpath + '/' + image,'JPEG')

# final resized 
final_imlist = [f for f in os.listdir(imgresizedpath) if os.path.splitext(f)[-1] == '.jpg']
imnum = len(final_imlist) #number of images

#size of 1 resized img
im1 = array(Image.open(imgresizedpath+'/'+final_imlist[0]))
m,n = im1.shape[0:2] 

#%%
# flatten the images
#imgflattened = array([array(Image.open(imgresizedpath + '/' + ime)).flatten()
#    for ime in final_imlist],'f')
    
input_data, final_imgclasses = shuffle(final_imgs, final_imgclasses , random_state=2 ) 
testtrain_data = (input_data,final_imgclasses)

#%%

(X,y) = (testtrain_data[0],testtrain_data[1])

# split testtrain_data into training and test set
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0],1,img_y,img_x)
X_test = X_test.reshape(X_test.shape[0], 1, img_y,img_x)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

## but we use RGB, how to change this?
X_train /= 255
X_test /= 255

###: convert the class vectors into binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)