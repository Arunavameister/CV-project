# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 06:33:15 2017

@author:ishwar

WARNING!! don't use for final submission
copied parts from here http://learnandshare645.blogspot.fr/2016/06/feeding-your-own-data-set-into-cnn.html
"""  

#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD,RMSprop,adam
#from keras.utils import np_utils

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

xmlpath = "C:/Users/edmon_000/Desktop/mldmyear2/computer vision/VOCdevkit/VOC2007/Annotations/"
imgpath = "C:/Users/edmon_000/Desktop/mldmyear2/computer vision/VOCdevkit/VOC2007/JPEGImages/"
#xmlpath = "C:/Users/edmon_000/Desktop/VOCdevkit_train/VOC2007/Annotations/"
#imgpath = "C:/Users/edmon_000/Desktop/VOCdevkit_train/VOC2007/JPEGImages/"

#imgpath = 'C:/Users/edmon_000/Desktop/mldmyear2/computer vision/VOCdevkit/VOC2007/JPEGImages/'
imgresizedpath = 'C:/Users/edmon_000/Desktop/mldmyear2/computer vision/VOCdevkit/VOC2007/input_data_pre_proprocessed/'

allclassfiles = glob.glob(xmlpath+'*.xml')

img_x = 200
img_y = 200

#dictionary of num and class
classes ={1:'aeroplane',
        2:'bicycle',
        3:'bird',
        4:'boat',
        5:'bottle',
        6:'bus',
        7:'car',
        8:'cat',
        9:'chair',
        10:'cow',
        11:'diningtable',
        12:'dog',
        13:'horse',
        14:'motorbike',
        15:'person',
        16:'pottedplant',
        17:'sheep',
        18:'sofa',
        19:'train',
        20:'tvmonitor'}


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
badimages = [] 

new_dict = []

i = 0

# resizes and stores images if the jpg images are found in the dictionary images_dict above
for image in alloriginals:
    #now check if the jpg file we are reading exists in the images_dict, if so  lets create new lists   
    # :-4 to remove the .jpg 
    imagenum = int(image[:-4])
    try:
        images_dict[imagenum]
        
        #first add the key to new list
        #final_imgnums.append(imagenum)
        
        #then add it's corresponding class retrieved from the xml dictionary
        final_imgclasses.append(images_dict[imagenum])
        final_imgs.append(i)
        new_dict.append(imagenum)
        i += 1
        
        #now save the good resized images (ones which are in the dictionary)
        #im = Image.open(imgpath + '/' + image)
        #img = im.resize((img_y,img_x))
        
        # this finally becomes a matrix of the dimension (number of images) * (img_x * img_y)
        #final_imgs.append(img)
        #img.save(imgresizedpath + '/' + image,'JPEG')
    except KeyError:
        badimages.append(image)


#test_with_labels = list(map(list,zip(final_imgs,final_imgclasses)))
#np.save(imgpath+"test_labels",test_with_labels)
#test_dic = dict(zip(new_dict, final_imgclasses))
#train_dic = dict(zip(new_dict, final_imgclasses))
#np.save(imgpath+"train_labels_dict",train_dic)

test_dic = dict(zip(new_dict, final_imgclasses))
np.save(imgpath+"test_labels_dict",test_dic)


#train_labels = list(map(list,zip(final_imgs,final_imgclasses)))
#np.save(imgpath+"train_val_labels",train_labels)
#train_dic = dict(zip(new_dict, final_imgs))
#np.save(imgpath+"train_val_labels_dict",train_dic)