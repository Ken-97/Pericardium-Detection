# -*- coding: utf-8 -*-
import numpy as np
import cv2
import png
import pydicom
from dicom import setDicomWinWidthWinCenter
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(threshold=np.inf)


for i in range(76,132):
    # read the dicom file
    if i >= 100:
        Patient = 'E:/cases/houyushan/_80940__0531111857/ser005img00' + str(i) + '.dcm'
        fileNmae = "ser005img00"+str(i)
    else:
        Patient = 'E:/cases/houyushan/_80940__0531111857/ser005img000' + str(i) + '.dcm'
        fileNmae = "ser005img000" + str(i)
    fileRoot = Patient
    dataset = pydicom.dcmread(fileRoot)
    dataset.image = dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept
    window_center = 40
    window_width = 400
    lung_my = setDicomWinWidthWinCenter(dataset.image, window_width, window_center, 512, 512)
    lung_cv = cv2.convertScaleAbs(lung_my, beta=0,alpha=(255.0 /window_width))
    img = lung_cv.copy()
    #在这一步中使用canny算法锁定腹腔的边缘。

    canny_edge = cv2.Canny(img, 50, 150)
    # cv2.imshow("canny_edge", canny_edge)

    #寻找上边界
    upper_edge_row = 0
    upper_edge_col = 0
    find = False
    for j in range(0,512):
        for k in range(0,512):
            if canny_edge[j,k] != 0:
                upper_edge_row = j
                upper_edge_col = k
                find = True
                break
        if find:
            break

    #寻找下边界
    bottom_edge_row = 0
    bottom_edge_col = 0
    find = False
    for j in range(511,-1,-1):
        for k in range(511,-1,-1):
            if canny_edge[j,k] !=0:
                bottom_edge_row = j
                bottom_edge_col = k
                find = True
                break
        if find:
            break

    bottom_edge_row_tmp = bottom_edge_row
    black_point = 0
    second_white = False
    for k in range(bottom_edge_row,-1,-1):
        if canny_edge[k,bottom_edge_col] == 0 and black_point == 0:
            black_point = 1
        if black_point == 1:
            if canny_edge[k,bottom_edge_col] != 0:
                second_white = True
        if second_white:
            if canny_edge[k,bottom_edge_col] == 0:
                black_point = 2
        if black_point == 2:
            if canny_edge[k,bottom_edge_col] != 0:
                bottom_edge_row = k
                break

    #在这里可以得到腹腔的整体高度，选取一个合适的比例，确定心脏所在的大致高度
    height_of_chest = bottom_edge_row - upper_edge_row
    heart_row_center =upper_edge_row + np.int(height_of_chest/3)
    # print heart_row_center
    #目测心脏宽度大致为130~140,高度大致100
    heart_col = 150 #130
    heart_row = 150 #100
    #此处假定心脏位于x轴中心位置，截取心脏部分

    print heart_row_center
    heart_img = img[heart_row_center-75:heart_row_center+75, 181:331]
    # cv2.line(canny_edge,(upper_edge_col,upper_edge_row),(bottom_edge_col,bottom_edge_row),255,5)

    # cv2.imshow("oriImg",img)
    # cv2.imshow("heart_img", heart_img)

    cv2.imwrite("E:/cases/houyushan/_80940__0531111857/_"+fileNmae+".bmp", heart_img)
    # print "imwrite finished."