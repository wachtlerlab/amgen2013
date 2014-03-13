#!/usr/bin/env python
# WCS array
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import colors

import argparse

def build_wcs_map():
	fd = open('cnum-vhcm-lab-new.csv')
	wcs_csv = csv.reader(fd, delimiter = '\t')
	wcs_map = { }
	for row in wcs_csv:
		if row[0][0] == '#':
			continue

		chip = { 'number' : row[0],
			 'hue' : row[4],
			 'chroma' : int(row[3]),
			 'value' : float(row[5]),
			 'Lab' : [float(6), float(7), float(8)] }
		index = "%s%s" % (row[1], row[2])
		wcs_map[index] = chip
#	print wcs_map
	return wcs_map

munselLetterMap = {
  'A' : 2.5,
  'B' : 5,
  'C' : 7.5,
  'D' : 10,
  'E' : 1.25,
  'F' : 3.75,
  'G' : 6.25,
  'H' : 8.75
}

def build_chiplist():
	fd = open('chips_index.csv')
	lines = fd.readlines()

	allchips = []	

	for idx,line in enumerate(lines):
		if line.startswith('NEUT'):
			chip = { 'hue' : 'NEUT',
			 'value' : 0,
			  'chroma' : 0,
			  'index' : idx }
			allchips.append(chip)
			continue

		hue_spec = line[0]
		hue = line[1:3]
		value = int(line[3:5]) / 10.0

		if hue[0] == hue[1]:
			hue = hue[0]

		hue_str = '%3.2f%s' % (munselLetterMap[hue_spec], hue)		
		
		chroma = int(line[5:7])
		chip = { 'hue' : hue_str,
			 'value' : value,
			  'chroma' : chroma,		   
			  'index' : idx  }
		allchips.append(chip)
	return allchips

def lookup_chip(hue, value, chroma, chipmap, fallback=True, fallthrough=False):
	
	if chroma < 0:
		return None
	
	for chip in chipmap:
		if chip['value'] == value and \
		   chip['hue'] == hue and \
	           chip['chroma'] == chroma:
			return chip
	
	#reaching this line means we didn't find a chip with hue, value, chroma
	if fallback:
		chip = lookup_chip(hue, value, chroma - 2, chipmap, fallthrough, fallthrough)  
	return chip

def create_chroma_grid(wcs_map, chips, args_chroma):
	empty = np.empty((8, 40))
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
			empty[row -1, col -1] = chip['chroma']
	return empty


def main():

	parser = argparse.ArgumentParser(description='Draw the WCS grid')
	parser.add_argument('--chroma', type=int, default=-1)
	args = parser.parse_args()

	wcs_map = build_wcs_map()
	#print wcs_map
	chips = build_chiplist()
	#print chips
	wcs_chip = wcs_map['C2']
	#chip = lookup_chip(wcs_chip['hue'], wcs_chip['value'], wcs_chip['chroma'], chips)
	#print 'C2', chip
	grid =  create_chroma_grid(wcs_map, chips, args.chroma) #np.empty((8, 40))					
	pic = plt.imshow(grid, interpolation='nearest')
	plt.gca().invert_yaxis()
	plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
	cbar = plt.colorbar(pic, shrink=0.25)
	cbar.set_ticks(xrange(2, 18, 2))
	plt.show()

if __name__ == '__main__':
	main()
