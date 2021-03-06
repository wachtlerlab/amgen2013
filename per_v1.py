#Project: Human perception of colors

import numpy as np
import scipy as sp
import math
import csv
import matplotlib.pyplot as plt
from numpy import *
import sys
import argparse

from munsell_wcs import build_wcs_map, build_chiplist, lookup_chip, create_chroma_grid

# importing daylight spectra [illumination]

daylight_data = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/baso4.asc', delimiter = '') # daylight illumination from reference white

daylight_data_1 = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/tree.asc', delimiter = '') # daylight illumination from spruce tree

daylight_data_2 = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/sky.asc', delimiter = '') # daylight illumination from the sky reflected by the mirror

# importing Munsell chips (Munsell380 glossy) [reflectance]

reflectance_data = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/mglossy_all/munsell380_780_1_glossy.asc', delimiter = '') # reflectance spectra of 1600 glossy Munsell color chips, 

#print reflectance_data
#print reflectance_data.shape

# importing cone sensitivities [cone sensitivities]

sensitivity_data = np.genfromtxt('/home/aurimas/Amgen/perception/linss2_10e_1.csv', delimiter = ',') # cones sensitivities
#print sensitivity_data[0:388:4,:]

baso4 = daylight_data.T;
tree = daylight_data_1.T;
sky = daylight_data_2.T;
print '-' * 10
#print baso4.shape, tree.shape, sky.shape

#for k in range(15):
#	print tree[0:97,k]

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
	    #print np.dot(sensitivity_data[0:388:4,i+1],tree[0:97,k])
	    #print np.multiply(sensitivity_data[0:388:4,i+1],tree[0:97,k]).shape
            U0[i,k] = np.sum(np.multiply(sensitivity_data[0:388:4,i+1],tree[0:97,k]))
            U1[i,k] = np.sum(np.multiply(sensitivity_data[0:388:4,i+1],baso4[0:97,k]))
            V0[i,k] = np.sum(np.multiply(np.multiply(sensitivity_data[0:388:4,i+1],reflectance_data[10:398:4,x]),tree[0:97,k]))
            V1[i,k] = np.sum(np.multiply(np.multiply(sensitivity_data[0:388:4,i+1],reflectance_data[10:398:4,x]),baso4[0:97,k]))

    U2 = zeros((3,22)) # defining U2 matrix
    V2 = zeros((3,22)) # defining V2 matrix

    for l in range(22):

        for j in range(3):

            U2[j, l] = np.sum(np.multiply(sensitivity_data[0:388:4,j+1],sky[0:97,l]))
            V2[j, l] = np.sum(np.multiply(np.multiply(sensitivity_data[0:388:4,j+1],reflectance_data[10:398:4,x]),sky[0:97,l]))

    U = np.concatenate((U0,U1,U2),axis=1) # mixing three U matrixes wrt columns: 3x52
    V = np.concatenate((V0,V1,V2),axis=1) # mixing three V matrixes wrt columns: 3x52
    #print U.shape
    #print U.size, V.size
    #print V[0,:]
    U_pinv = np.linalg.pinv(U) # finding pseudoinverse matrix of U 

    A[:,:,x] = np.dot(V,U_pinv) # finding linear operator [A] that satisfies equation V = AU

    D,E = linalg.eig(A[:,:,x]); # eigendecomposition (of linear operator, D - eigenvalues, E - eigenvectors (represent diffent hue compoisitions))

    if D.dtype == np.complex128:  
         #c.writerow(D)
         D=D.real
         #print D
    #print D

    B1[x] = abs (D[0]) / abs(D[1]) # special case 1: first type of singularity - variation in incoming light produces strong variation in the reflected light along one direction
    #print B1[x]
    B2[x] = abs(D[1]) / abs(D[2]) # special case 2: econd type of singularity - variation in incoming light only produces strong variations in the reflected light along two directions
 
b1 = B1.max()
b1_1 = B1.min() #corresponds to surface that reflect one of the incoming light elements more significantly than the other two (aka strong variation of the reflected light along one direction)
#print b1

b2 = B2.max() #corresponds to surface that reflect two incoming light elements much more significantly than the third (aka strong variation of the reflected light along two directions)
#print b2

B = zeros(1600) # defining B matrix  

for x in range (1600):

    B[x] = max(B1[x]/b1,B2[x]/b2) # finding each singularity index

#print V.shape
#print U.shape
#print U_pinv.shape
#print A.shape
#print D.shape
#print E.shape
#print sp.diag((D))
#print B[x]

wcs_map = build_wcs_map()
chips = build_chiplist()
wcs_chip = wcs_map['C2']
#chip = lookup_chip(wcs_chip['hue'], wcs_chip['value'], wcs_chip['chroma'], chips)
#print 'C2', chip

parser = argparse.ArgumentParser(description='Calculate Singularity')
parser.add_argument('--chroma', type=int, default=-1)
parser.add_argument('--grid', action='store_true', default=False)
args = parser.parse_args()


hue_x = np.empty((8, 40))

for row in xrange(1, 9):
	R = chr(ord('A') + row)
	for col in xrange(1, 41):
		idx = '%s%d' % (R, col)
		wcs_chip = wcs_map[idx]
		if args.chroma > 0:
			chroma = args.chroma
			do_fallback = True
			do_fallthrough = True
		else:
			chroma = wcs_chip['chroma']
			do_fallback = True
			do_fallthrough = False

		chip = lookup_chip(wcs_chip['hue'], wcs_chip['value'], chroma, chips, fallback=do_fallback, fallthrough=do_fallthrough)
#		print idx, chip['index'], B[chip['index']]
		hue_x[row-1, col-1] = B[chip['index']]

fig = plt.figure()
ax = fig.add_subplot([1,2][args.grid],1,1)
plt.imshow(hue_x, interpolation='nearest')
chroma_str = ('WCS', '%d' % args.chroma)
ax.set_title('Singularity for chroma %s' % chroma_str[args.chroma > 0])
plt.gca().invert_yaxis()
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
cbar = plt.colorbar(shrink=[0.27, 0.5][args.grid])
#cbar.set_ticks(xrange(np.min(B[:]), np.max(B[:]), ))

if args.grid:
	ax = fig.add_subplot(2,1,2)
	ax.set_title('Chroma grid')
	grid =  create_chroma_grid(wcs_map, chips, args.chroma) #np.empty((8, 40))					
	pic = plt.imshow(grid, interpolation='nearest')
	plt.gca().invert_yaxis()
	plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
	cbar = plt.colorbar(pic, shrink=0.5)
	cbar.set_ticks(xrange(2, 18, 2))

plt.show()




