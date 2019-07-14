# -*- coding: utf-8 -*-
import numpy as np
import cv2
import png
import pydicom
from dicom import setDicomWinWidthWinCenter
from matplotlib import pyplot as plt

matrix1= np.ones([400,400],np.uint8)
matrix2 = np.ones([402,402],np.uint8)
paddingCol=np.zeros(400,np.uint8)
paddingRow=np.zeros(402,np.uint8)
outcome = np.c_[paddingCol,matrix1,paddingCol]
outcome = np.r_[[paddingRow], outcome, [paddingRow]]
matrix2[outcome == 1] = 0
cv2.imshow("test", matrix2)
print matrix2
cv2.waitKey()