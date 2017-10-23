# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:24:51 2017

@author: ishwar
"""

import glob
import xml.etree.ElementTree
import re


#this is the path to the XML directory of all the images
#FIRST delete all the merde mac files before


#### change path according to your computer
path = "C:/Users/edmon_000/Desktop/mldm year 2/computer vision/VOCdevkit/VOC2007/Annotations/"
allclassfiles = glob.glob(path+'*.xml')


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

for f in allclassfiles:
    e = xml.etree.ElementTree.parse(f).getroot()
    imgnum = e[1].text
    imgnum = re.sub('\.jpg$','',imgnum)
    print(imgnum + " " + e[6][0].text + " " + str(classes.get(e[6][0].text)))