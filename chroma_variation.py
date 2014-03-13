#!/usr/bin/env python
# Spectra change when varying chroma (chroma_variation.py #chip); screen resolution provides chip colors that hardly correspond to unique hues
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec

import argparse
from munsell_wcs import build_wcs_map, build_chiplist, lookup_chip, create_chroma_grid
from colormath.color_objects import LabColor

def wcs_gen_chip(chip):
	lab = LabColor(*chip['Lab'])
	rgb = lab.convert_to('rgb', debug=False, illuminant='d50')
	img = np.ones((100, 100, 3))	
	img[:,:,0] *= rgb.rgb_r / 255.0
	img[:,:,1] *= rgb.rgb_g / 255.0
	img[:,:,2] *= rgb.rgb_b / 255.0
	return img

def gen_spectra(chip, chip_list):
	path = os.path.expanduser('~/Amgen/amgen_2013/data/lut.fi/mglossy_all/munsell380_780_1_glossy.asc')
	reflectance_data = np.genfromtxt(path, delimiter = '')

	chroma_start = chip['chroma']
#	print chroma_start
	chromas = np.arange(chroma_start, 0, -2) #[10, 8, 6 .. 2]
	nchroma = len(chromas) # number in the list, ie chromas
	spectra = np.empty((reflectance_data.shape[0], nchroma)) # 401, nchroma 

	for idx, chroma in enumerate(chromas):
		cur_chip = lookup_chip(chip['hue'], chip['value'], chroma, chip_list)	
		cur_spectrum = reflectance_data[:, cur_chip['index']]
		spectra[:,idx] = cur_spectrum
	return spectra, chromas

def main():
	parser = argparse.ArgumentParser(description='Varying chroma')
	parser.add_argument('wcs_coordinate', type=str)

	args = parser.parse_args()

	chips = build_chiplist()
	wcs_map = build_wcs_map()

	coord = args.wcs_coordinate
	wcs_chip = wcs_map[coord]
	
	chip = lookup_chip(wcs_chip['hue'], wcs_chip['value'], wcs_chip['chroma'], chips)
	spectra, chromas = gen_spectra(chip, chips)
	fig = plt.figure()
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3]) 
	ax = fig.add_subplot(gs[1])
	x = np.arange(0, spectra.shape[0]) + 380	
	plt.plot(x, spectra)
	plt.ylim([0, 1.0])
	plt.xlim([380, 780])
	ax.set_title('WCS chip: %s' % coord)
	plt.legend([str(x) for x in chromas],prop={'size':8})
	ax = fig.add_subplot(gs[0])

	ax.set_xticks([])
	ax.set_yticks([])
	img = wcs_gen_chip(wcs_chip)
	fig.set_dpi(200)
#	plt.savefig('G30.eps',dpi=300)	
	plt.imshow(img)
	plt.show()
	

if __name__ == '__main__':
	main()
