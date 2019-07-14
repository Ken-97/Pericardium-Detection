# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pydicom
from dicom import setDicomWinWidthWinCenter
from matplotlib import pyplot as plt

'''
dataset = pydicom.dcmread("E:/cases/_80940__0531100918/ser005img00001.dcm")
shape = dataset.pixel_array.shape
#convert to float to avoid overflow or underflow losses
image_2d = dataset.pixel_array.astype(float)

#rescaling grey scale between 0-255
image_2d_scaled = (np.maximum(image_2d,0)/image_2d.max())*255.0

#convert to unit
image_2d_scaled = np.uint8(image_2d_scaled)

#Write the png file
with open("E:/cases/_80940__0531100918/ser005img00001.png", 'wb') as png_file:
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(png_file, image_2d_scaled)

'''''

input_dir = 'E:/cases'

for i in range(50,51):
    Patient = input_dir + '/houyushan/_80940__0531111857/ser005img000' + str(i) + '.dcm'
    fileRoot = Patient
    dataset = pydicom.dcmread(fileRoot)
    dataset.image = dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept
    window_center = 60
    window_width = 400
    #40,400
    #这里不知道是哪一步出了问题，转化的图像总是比软件转化的要暗
    # cv2.imshow("dataset.image", dataset.image)
    lung_my = setDicomWinWidthWinCenter(dataset.image, window_width, window_center, 512, 512)
    # cv2.imshow("lung_my", lung_my)
    #lung_cv = cv2.convertScaleAbs(dataset.image-window_center, alpha=(255.0 /window_width))
    #lung_cv = cv2.convertScaleAbs(lung_my, beta=abs(window_center-0.5*window_width),alpha=(255.0 /window_width))
    lung_cv = cv2.convertScaleAbs(lung_my, beta=0,alpha=(255.0 /window_width))

    img = lung_cv

    # print img

    #img = cv2.imread("E:/cases/_80940__0531100918/ser005img00001.png")

    cv2.imshow("original image", img)
    #
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # cv2.imshow("gray", gray)

    ret, thresh = cv2.threshold(img,26,255,cv2.THRESH_BINARY)
    #26

    cv2.imshow("threshold 30", thresh)

    # 0中文
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)

    cv2.imshow("opening", opening)

    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    cv2.imshow("sure background", sure_bg)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L1,3)

    # cv2.imshow("dist_transform", dist_transform)

    ret,sure_fg = cv2.threshold(dist_transform, 0.01*dist_transform.max(),255,0)

    # cv2.imshow("sure foreground", sure_fg)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # cv2.imshow("subtraction", unknown)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    markers = cv2.watershed(img, markers)
    markers_backup = markers
    img[markers == -1] = [0,0,0]
    img[markers != -1] = [255,255,255] #注意，在这里marker其实就已经是轮廓的坐标集合了。但是这个坐标集合只有点，并没有很仔细地区分出一个个轮廓，所以是不是和直接使用drawcontours的。
    cv2.imshow("img with markers information", img)
    ##########################################1101
    # 漫水法上色
    mask = np.zeros([514,514], np.uint8) #初始化
    paddingCol = np.zeros(512)
    paddingRow = np.zeros(514)
    markers = np.c_[paddingCol, markers, paddingCol]
    markers = np.r_[[paddingRow], markers, [paddingRow]]
    mask[markers == -1] = 1;

    cv2.floodFill(img, mask,(1,1),(0,0,0),loDiff=(10,10,10),upDiff=(20,20,20),flags=cv2.FLOODFILL_FIXED_RANGE)
    # cv2.imshow("mask", mask)
    # cv2.imshow("afterFloodFill", img)

    #反色处理
    for j in range(512):
        for k in range(512):
            img[j,k] = (255 - img[j,k])

    inner = False
    for j in range(512):
        for k in range(512):
            if markers_backup[j,k]==-1 and inner==False:
                inner=True
                if j-1>0 and k-1>0:
                    img[j,k]=[0,0,0]
            if markers_backup[j,k]==-1 and inner==True:
                inner=False
                continue

    cv2.imshow(str(i), img)


    ############################1101
    #

    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # findcontours modifies the source image


    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = np.uint8(img)
    # im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("contours found",im2)
    # c_normal = []
    # for j in range(len(contours)):
    #     cnt = contours[j]
    #     area = cv2.contourArea(cnt)
    #     if(area < 200*200):
    #         c_normal.append(cnt)
    # tmp = np.zeros_like(img)
    # for i in range(len(c_normal)):
    #     tmp = cv2.drawContours(tmp,c_normal,i,255,-1)
    #     cv2.imshow("afterDrew", tmp)
    #     cv2.waitKey(200)
    # cv2.imshow("markers", markers)



    # flag = 0
    #
    # for i in np.arange(512):
    #     for j in np.arange(512):
    #         if (img[i,j]==[0,0,0]).all():
    #             flag = 1
    #         elif (flag == 1) and (img[i,j]==[0,0,0]).all():
    #             flag = 2
    #
    #         if flag == 1 and (img == [255,255,255]).all():
    #             img = [0,0,0]
    #     flag = 0


    # cv2.imshow(str(i), im2)
    cv2.waitKey(20)

cv2.waitKey()