#Project: Human perception of colors v3

import numpy as np
import scipy as sp
import math
import csv
import numpy.ma as ma
import matplotlib.pyplot as plt
from numpy import *
import sys

# importing daylight spectra [illumination]

daylight_data = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/baso4.asc', delimiter = '') # daylight illumination from reference white

daylight_data_1 = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/tree.asc', delimiter = '') # daylight illumination from spruce tree

daylight_data_2 = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/sky.asc', delimiter = '') # daylight illumination from the sky reflected by the mirror

# importing Munsell chips (Munsell380 glossy) [reflectance]

reflectance_data = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/mglossy_all/munsell380_780_1_glossy.asc', delimiter = '') # reflectance spectra of 1600 glossy Munsell color chips, 

# importing cone sensitivities [cone sensitivities]

sensitivity_data = np.genfromtxt('/home/aurimas/Amgen/perception/linss2_10e_1.csv', delimiter = ',') # cones sensitivities

baso4 = daylight_data.T;
tree = daylight_data_1.T;
sky = daylight_data_2.T;
print '-' * 10

A = zeros((3,3,1600,)) # defining A matrix
B1 = zeros(1600) # defining B1 matrix
B2 = zeros(1600) # defining B2 matrix

c = csv.writer(open("eigenvalues.csv", "w"))

for x in range(1600):
    
    U0 = zeros((3,15)) # defining U0 matrix
    U1 = zeros((3,15)) # defining U1 matrix
    V0 = zeros((3,15)) # defining V0 matrix
    V1 = zeros((3,15)) # defining V1 matrix

    for k in range(15):

        for i in range(3):

            U0[i,k] = np.sum(np.multiply(sensitivity_data[0:388:4,i],tree[0:97,k]))
            U1[i,k] = np.sum(np.multiply(sensitivity_data[0:388:4,i],baso4[0:97,k]))
            V0[i,k] = np.sum(np.multiply(sensitivity_data[0:388:4,i],reflectance_data[10:398:4,x],tree[0:97,k]))
            V1[i,k] = np.sum(np.multiply(sensitivity_data[0:388:4,i],reflectance_data[10:398:4,x],baso4[0:97,k]))

    U2 = zeros((3,22)) # defining U2 matrix
    V2 = zeros((3,22)) # defining V2 matrix

    for l in range(22):

        for j in range(3):

            U2[j, l] = np.sum(np.multiply(sensitivity_data[0:388:4,j],sky[0:97,l]))
            V2[j, l] = np.sum(np.multiply(sensitivity_data[0:388:4,j],reflectance_data[10:398:4,x],sky[0:97,l]))
 
    U = np.concatenate((U0,U1,U2),axis=1) # mixing three U matrixes wrt columns: 3x52
    V = np.concatenate((V0,V1,V2),axis=1) # mixing three V matrixes wrt columns: 3x52
    
    U_pinv = np.linalg.pinv(U) # finding pseudoinverse matrix of U 

    A[:,:,x] = np.dot(V,U_pinv) # finding linear operator [A] that satisfies equation V = AU

    D,E = linalg.eig(A[:,:,x]); # eigendecomposition (of linear operator, D - eigenvalues, E - eigenvectors (represent diffent hue compoisitions))
    #print D

    if D.dtype == np.complex128:  
        c.writerow(D)

    B1[x] = abs (D[0]) / abs(D[1])
    #print B1[x]
    B2[x] = abs(D[1]) / abs(D[2])
    B11 = ma.array(B1[x],mask=zeros,fill_value=None,keep_mask = False) #?
    
print B11

b1 = B1[x].max()
print b1

b2 = B2[x].max()
print b2

B = zeros(1600) # defining B matrix  

for x in range (1600):

    B[x] = max(B1[x]/b1,B2[x]/b2) # finding each singularity index

print V.shape
print U_pinv.shape
print A.shape
print D.shape
print E.shape
print sp.diag((D))
print B[x]
