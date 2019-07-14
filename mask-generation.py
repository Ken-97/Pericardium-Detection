#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
import numpy as np
from PIL import Image
import cv2
import time
import csv
import pydicom

if __name__ == '__main__':
	for index in range (76,77):
		red_mask = np.zeros((512,512))
		green_mask = np.zeros((512,512))
		blue_mask = np.zeros((512,512))
		file_root = "/home/ranyang1/Pericardium-Project/data/data/Train"
		if index < 100:
			file_name = "_ser005img000" + str(index) + ".bmp"
		else:
			file_name = "_ser005img00" + str(index) + ".bmp"
		file_root = file_root + file_name
		file = Image.open(file_root)
		for i in range(0, np.size(file, 0)):
			for j in range(0,np.size(file,1)):
				pixel = file.getpixel((i,j))
				if (pixel[0] > pixel[1] and pixel[0] > pixel[2]):
					red_mask[j,i] = 255
				if (pixel[1] > pixel[0] and pixel[1] > pixel[2]):
					green_mask[j,i] = 255
				if (pixel[2] > pixel[0] or pixel[2] > pixel[1]):
					blue_mask[j,i] = 255
		red_mask_image = Image.fromarray(red_mask)
		red_mask_image = red_mask_image.convert("L")
		green_mask_image = Image.fromarray(green_mask)
		green_mask_image = green_mask_image.convert("L")
		blue_mask_image = Image.fromarray(blue_mask)
		blue_mask_image = blue_mask_image.convert("L")

		red_mask_image.save("red-mask.bmp")
		green_mask_image.save("green-mask.bmp")
		blue_mask_image.save("blue-mask.bmp")
