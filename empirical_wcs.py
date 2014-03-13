#!/usr/bin/env python

# Speaker frequency on each chip in the WCS

import os
import numpy as np
print np.__version__
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from munsell_wcs import build_wcs_map, build_chiplist, lookup_chip
from collections import Counter
from itertools import izip

# Counting and transfering to csv file 

hue = np.genfromtxt('/home/aurimas/Amgen/perception/chips_and_speakers.csv', delimiter =',', usecols=(0),dtype=None)
hue_value = np.genfromtxt('/home/aurimas/Amgen/perception/chips_and_speakers.csv', delimiter =',', usecols=(1))
speakers = np.genfromtxt('/home/aurimas/Amgen/perception/chips_and_speakers.csv', delimiter =',', usecols=(2))

x = hue
y = hue_value
z = speakers

def create_file():
	with open('foci-exp.csv', 'rU') as emp:
		emp_csv = csv.reader(emp, delimiter = '\t')
		emp_map = { }
		cnt = Counter()
		for row in emp_csv:
			chip_count = [row[4]]
			for word in chip_count:
				cnt[word] += 1
			chips_cas = cnt
		with open('chips_and_speakers_sorted.csv', 'wb') as f:
			w = csv.writer(f, delimiter=',')
			for c in chips_cas.iteritems():
				w.writerow([c[0][0], c[0][1:], c[1]])

a = izip(*csv.reader(open("chips_and_speakers_sorted.csv", "rb")))
csv.writer(open("chips_and_speakers0.csv", "wb")).writerows(a)

	#print chips_cas.values()
	#print chips_cas.keys()
	#sort = sorted(chips_cas)
	#hueaval = sorted(hues_and_values)
	#speaks = sorted(speak)
	#print hueaval, speak
#print hue, hue_value, speakers

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

#fig = plt.figure()
#ax = fig.gca(projection='3d')

#Axes3D.plot(hue_value,hue, speakers)
hue_num = [(ord(X) - ord('A')) for X in hue]

tosortlist = zip(hue_num, hue_value, speakers)
sortedlist = sorted(tosortlist)
unzipedlist = zip(*sortedlist)

hue_num = np.array(unzipedlist[0])
hue_value = np.array(unzipedlist[1])
speakers = np.array(unzipedlist[2])


#print hue_num
X = hue_num.reshape((10, 41))
Y = hue_value.reshape((10, 41))
Z = speakers.reshape((10, 41))

wcs_map = build_wcs_map()
chips = build_chiplist()
wcs_chip = wcs_map['C2']
chip = lookup_chip(wcs_chip['hue'], wcs_chip['value'], wcs_chip['chroma'], chips)
#print 'C2', chip
empty = np.empty((8, 40))
for row in xrange(1, 9):
	R = chr(ord('A') + row)
	for col in xrange(1, 41):
		idx = '%s%d' % (R, col)
		wcs_chip = wcs_map[idx]
		chip = lookup_chip(wcs_chip['hue'], wcs_chip['value'], wcs_chip['chroma'], chips)
#			print idx, chip['index']
		empty[row -1, col -1] = chip['chroma']
Z_new = Z[1:9, 1:41] / empty
Z = Z[1:9, 1:41]
#print Z_new.shape

#plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
plt.imshow(np.flipud(Z_new), interpolation='nearest', label='B&K')
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']) # speakers frequancy with normalisation
#plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']) #involving achromatic hues, no normalisation
plt.colorbar(shrink=0.26)
plt.title('Speakers frequency')
#plt.gca().invert_yaxis()
plt.savefig('Speakers in 6.eps', dpi=600, figsize=(8, 6))
plt.show()
#plt.set_title('Total speaker number for Munsell chip')
