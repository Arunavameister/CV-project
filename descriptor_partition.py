# -*- coding: utf-8 -*-
"""
gets the feature vectors from matlab clusters and descriptors
@author: edmon_000
"""

import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.spatial.distance as distFunc
from sklearn.metrics.pairwise import euclidean_distances

dirpath = "C:/Users/edmon_000/Desktop/mldmyear2/computer vision/testruns/matlab_output_700/"

# read all the 700 clusters x 128 dimensions
clusters =  np.transpose(sio.loadmat(dirpath+'clusters_700.mat')['centers'])
num_clusters = clusters.shape[0]


#numfiles = 4952
numfiles = 4952
partition_all = np.zeros((numfiles,num_clusters))

imgname = "JPEGImages"


for i in range(numfiles):
    filename = imgname+str(i+1)
    descriptor =  np.transpose(sio.loadmat(dirpath+filename)['desc'])
    distall = euclidean_distances(descriptor,clusters)
    #now calculate euclid distance for each desc element
    num_descriptors = descriptor.shape[0]
    for j in range(num_descriptors):
        #cluster_index = np.argmin([distFunc.euclidean(descriptor[j],clusters[cc]) for cc in range (num_clusters)])
        cluster_index = np.argmin(distall[j])
       # print(cluster_index)
        partition_all[i,cluster_index] += 1 
# loop through each image descriptor of size 128

#a = np.asarray(partition_all)
header = [np.transpose([i+1 for i in range(num_clusters)])]
for_output = np.concatenate((header,  partition_all), axis=0).reshape(numfiles+1,num_clusters)
np.savetxt(dirpath+"partition_output.csv", for_output,fmt='%d' ,delimiter=",")

# for each image descriptor, get its distance from each cluster, and store it in a 700 size array 
#  --> cluster[np.argmin(a)] += 1 

               
               