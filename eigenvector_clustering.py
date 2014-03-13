#!/usr/bin/env python

import numpy as np
import scipy as sp
import math
import csv
import matplotlib.pyplot as plt
from numpy import *
import sys
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from colormath.color_objects import LabColor

import scipy.cluster.vq as cluster

from munsell_wcs import build_wcs_map, build_chiplist, lookup_chip, create_chroma_grid

def create_A_all():
	daylight_data = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/baso4.asc', delimiter = '') 
	daylight_data_1 = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/tree.asc', delimiter = '') 
	daylight_data_2 = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/daylight/sky.asc', delimiter = '') 
	reflectance_data = np.genfromtxt('/home/aurimas/Amgen/amgen_2013/data/lut.fi/mglossy_all/munsell380_780_1_glossy.asc', delimiter = '')
	sensitivity_data = np.genfromtxt('/home/aurimas/Amgen/perception/linss2_10e_1.csv', delimiter = ',') 
	baso4 = daylight_data.T;
	tree = daylight_data_1.T;
	sky = daylight_data_2.T;
	print '-' * 10

	A = zeros((3,3,1600,)) 
	B1 = zeros(1600)
	B2 = zeros(1600)

	for x in range(1600):
	    
	    U0 = zeros((3,15))
	    U1 = zeros((3,15))
	    V0 = zeros((3,15))
	    V1 = zeros((3,15))

	    for k in range(15):

		for i in range(3):
		    U0[i,k] = np.sum(np.multiply(sensitivity_data[0:388:4,i+1],tree[0:97,k]))
		    U1[i,k] = np.sum(np.multiply(sensitivity_data[0:388:4,i+1],baso4[0:97,k]))
		    V0[i,k] = np.sum(np.multiply(np.multiply(sensitivity_data[0:388:4,i+1],reflectance_data[10:398:4,x]),tree[0:97,k]))
		    V1[i,k] = np.sum(np.multiply(np.multiply(sensitivity_data[0:388:4,i+1],reflectance_data[10:398:4,x]),baso4[0:97,k]))

	    U2 = zeros((3,22))
	    V2 = zeros((3,22))

	    for l in range(22):

		for j in range(3):
		    U2[j, l] = np.sum(np.multiply(sensitivity_data[0:388:4,j+1],sky[0:97,l]))
		    V2[j, l] = np.sum(np.multiply(np.multiply(sensitivity_data[0:388:4,j+1],reflectance_data[10:398:4,x]),sky[0:97,l]))

	    U = np.concatenate((U0,U1,U2),axis=1)
	    V = np.concatenate((V0,V1,V2),axis=1)
	    U_pinv = np.linalg.pinv(U)
	    A[:,:,x] = np.dot(V,U_pinv)
	return A

def plot_data(A, color):
	nops = A.shape[2]
	E_all = np.empty((A.shape[0], A.shape[1]*A.shape[2])) # 3, 3*1600 or 3*320
	for idx in xrange(0, nops):
		D,E = linalg.eig(A[:,:,idx])
		#                  v--- cloumns end
		#	     v--- columns start
		#      v--- all rows
		E_all[ : , 3*idx:3*idx+3] = E # idx == 0 -> 0:3, idx == 1 -> 3:6 
	

#	c3, label = cluster.kmeans2(E_all.T, 3)
	c4, label = cluster.kmeans2(E_all.T, 6)

	print color.shape
	fig = plt.figure()
#	ax = fig.add_subplot(1, 1, 1, projection='3d') #1 2 2 1
#	ax.scatter(E_all[0,:], E_all[1,:], E_all[2,:], marker='s', c=color.T, edgecolor='none')
##	ax.scatter(c3[:,0], c3[:,1], c3[:,2], marker='x',  c='r')
#	ax.scatter(c4[:,0], c4[:,1], c4[:,2],  c='k', marker='x', s=60)
#	ax.set_xlabel("L",fontsize=18)
#	ax.set_ylabel('M',fontsize=18)
#	ax.set_zlabel("S",fontsize=18)
#	fig.savefig('CLUSTERS(1).eps', dpi=300)

#	ax = fig.add_subplot(1, 1, 1) 2 .. 2
#	ax.scatter(E_all[0,:], E_all[1,:], marker='s', c=color.T, edgecolor='none')
##	ax.scatter(c3[:, 0], c3[:, 1],  c='r')
#	ax.scatter(c4[:,0], c4[:,1],  c='k', marker='x', s=60)
#	ax.set_xlabel("L",fontsize=18)
#	ax.set_ylabel('M',fontsize=18)
#	fig.savefig('CLUSTERS(M vs L)c6_300.eps', dpi=300)

#	ax = fig.add_subplot(1, 1, 1) 3 .. 3
#	ax.scatter(E_all[1,:], E_all[2,:], marker='s', c=color.T, edgecolor='none')
##	ax.scatter(c3[:,1], c3[:,2],  c='r')
#	ax.scatter(c4[:,1], c4[:,2],  c='k', marker='x', s=60)
#	ax.set_xlabel("M",fontsize=18)
#	ax.set_ylabel('S',fontsize=18)
#	fig.savefig('CLUSTERS(S vs M)c6_300.eps', dpi=300)

	ax = fig.add_subplot(1, 1, 1) #4 .. 4
	ax.scatter(E_all[0,:], E_all[2,:], marker='s', c=color.T, edgecolor='none', lw=1)
#	ax.scatter(c3[:,0], c3[:,2],  c='r')
	ax.scatter(c4[:,0], c4[:,2],  c='k', marker='x', s=60)
	ax.set_xlabel("L",fontsize=18)
	ax.set_ylabel('S',fontsize=18)
	fig.savefig('CLUSTERS(S vs L)c6_300.eps', dpi=300)

def get_wcs_indicies(args_chroma):
	wcs_map = build_wcs_map()
	chips = build_chiplist()
	inidicies = np.empty((8*40))
	color_rgb = np.empty((8*40, 3))

	for row in xrange(1, 9):
		R = chr(ord('A') + row)
		for col in xrange(1, 41):
			idx = '%s%d' % (R, col)
			wcs_chip = wcs_map[idx]
			if args_chroma > 0:
				chroma = args_chroma
				do_fallback = True
				do_fallthrough = True
			else:
				chroma = wcs_chip['chroma']
				do_fallback = True
				do_fallthrough = False

			chip = lookup_chip(wcs_chip['hue'], wcs_chip['value'], chroma, chips, fallback=do_fallback, fallthrough=do_fallthrough)
			inidicies[(row-1) * 40 + col-1] = chip['index']
			lab = LabColor(*wcs_chip['Lab'])
			rgb = lab.convert_to('rgb', debug=False, illuminant='d50')
			color_rgb[(row-1) * 40 + col-1, :] = np.array([rgb.rgb_r, rgb.rgb_g, rgb.rgb_b])
	return (inidicies, color_rgb)

def main():	
	parser = argparse.ArgumentParser(description='Calculate Singularity')
	parser.add_argument('--all', action='store_true', default=False)
	args = parser.parse_args()
	A = create_A_all()
	print '-' * 10

	color = 'k'
	if not args.all:
		idcs, color = get_wcs_indicies(-1)
		idcs_int = [int(x) for x in idcs.flatten()]
		A = A[:,:,idcs_int]

	plot_data(A, color)
	plt.show()
if __name__ == '__main__':
	main()

