#!/usr/bin/env python

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from munsell_wcs import build_wcs_map, build_chiplist, lookup_chip, main
from collections import Counter
from itertools import izip
from matplotlib import colors

def constant_chroma():
	empty = np.empty((8, 40))
	for row in xrange(1, 9):
		R = chr(ord('A') + row)
		for col in xrange(1, 41):
			idx = '%s%d' % (R, col)
			wcs_chip = wcs_map[idx]
			chip = lookup_chip(wcs_chip['hue'], wcs_chip['value'], wcs_chip['chroma'], chips)
			empty[row -1, col -1] = chip['chroma']
			#print idx, chip['index']
				for row in xrange(1,9):
				
	plt.imshow(empty, interpolation='nearest')
	plt.gca().invert_yaxis()
	plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
	plt.show()
	#chipconstant = constant()
	#for chips in chipconstant:
		#if chip['chroma'] >= int(6):
		#	set([6])
		#else:
		#	return chip

if __name__ == '__constant_chroma__':
	constant_chroma()
