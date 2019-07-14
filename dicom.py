#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pydicom
import numpy
import math
import cv2

def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    img_temp = img_data.copy()
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

#'''
# Here we changed the formula into this one:
#y = 255 / (1 + e^(-x)) - 1/2 * winwidth
#,where y is the tranformed value and the x is the former value. This function refer to the sigmod function.
# '''


    # for i in numpy.arange(rows):
    #    for j in numpy.arange(cols):
    #         img_temp[i, j] = int((img_temp[i,j] - min)*dFactor)

    e = 2.71828182846
    pi = 3.14159265358979323846
    f1 = math.sqrt(2*pi)
    sd = 500.0
    u = 1000.0

    c = 1.0 / (sd * f1)
    for i in numpy.arange(rows):
        for j in numpy.arange(cols):

            img_temp[i, j]= -1.0*(pow((img_temp[i, j] - u), 2))/(2*sd*sd)
            img_temp[i, j]= int(255.0*pow(e, img_temp[i, j]))



            # if img_temp[i,j] >= wincenter - 0.5*winwidth and img_temp[i,j] <= wincenter + 0.5*winwidth:
            #     img_temp[i, j] = int((img_temp[i, j] - min) * dFactor)
            # else:
            #     img_temp[i, j] = int((img_temp[i, j] - min) * dFactor)

            # img_temp[i,j]=int(math.sin(math.radians(90)-winwidth+img_temp[i,j]))
            # print img_temp[i,j]


    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255
    return img_temp