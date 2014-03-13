#!/usr/bin/env python

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

	#speak = chips_cas.values()
	#hues_and_values = chips_cas.keys()
	#sort = sorted(chips_cas)
	#hueaval = sorted(hues_and_values)
	#speaks = sorted(speak)
	#print hueaval, speak
#print chips_cas, hue, hue_value, chroma

#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False,  cmap=cm.coolwarm)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#n = 100
#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#    xs = randrange(n, 23, 32)
#    ys = randrange(n, 0, 100)
#    zs = randrange(n, zl, zh)
#    ax.scatter(xs, ys, zs, c=c, marker=m)

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

plt.show()


