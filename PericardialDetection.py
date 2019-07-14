# -*- coding: utf-8 -*-
import numpy as np
import cv2
import png
import pydicom
from dicom import setDicomWinWidthWinCenter
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(threshold=np.inf)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])



input_dir_PD = 'E:/fromDiskE/Study/Ran.Y Project/Pericardium Detection/landmark/170x30/'

files_sum = np.zeros((30,170))
#
# aver_matrix = np.ones((30,170))
#
# aver_matrix = aver_matrix * 10;
for i in range(1,11):
    # files_normal = np.zeros((30,170))
    RetrosternalArea = input_dir_PD + str(i) + '.png'
    fileRoot = RetrosternalArea
    files = mpimg.imread(fileRoot)
    files = rgb2gray(files)
    files = ((files - files.min())*1.0)/((files.max() - files.min())*1.0)
    files[files < 0.2] = 0 #Here, those which density is less than 0.2 are filtered.
    files_sum = files_sum + files

files_sum = files_sum / 10

atlas = files_sum

tmp = np.uint8(files_sum)
# cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
cv2.imshow("atlas", files_sum)
cv2.waitKey(2)
# plt.imshow(files_sum)
# plt.show()

for i in range(104,111):
    # read the dicom file
    Patient = 'E:/cases/chenbingquan/_80940__0531100918/ser005img00' + str(i) + '.dcm'
    fileRoot = Patient
    dataset = pydicom.dcmread(fileRoot)
    dataset.image = dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept
    window_center = -115
    window_width = 170
    lung_my = setDicomWinWidthWinCenter(dataset.image, window_width, window_center, 512, 512)
    lung_cv = cv2.convertScaleAbs(lung_my, beta=0,alpha=(255.0 /window_width))
    img = lung_cv
    img = ((img - img.min()) * 1.0) / ((img.max() - img.min()) * 1.0)
    # print img
    # cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("test", img)
    # cv2.waitKey()

    # plt.imshow(img)
    # plt.show()

    #the coordinate of the upper left point of the moving image
    #REMEMBER: THE SIZE OF THE ATLAS IS 30X170, AND THAT OF THE
    #IMAGE IS 512X512, SO THE MAX SUBSCRIPT OF ROW IS 481 AND
    #THAT OF THE COLUMN IS 341.
    x_ul = -1
    y_ul = -1
    moving_x = 30
    moving_y = 170
    # Here we use the average of the img as the threshold to
    # define the bright and dark area.
    t = 0.02
    # t = 0.3
    # print t
    HDM = -float('inf')
    # print "HDM = " + str(HDM)

    # curr_fix = np.zeros((30,170))

    # plt.imshow(img)
    # plt.show()
    MD_final = float('inf')
    over_flag = False
    for j in range(0,481):
        for k in range(0, 341):
            #pick out the current fix image
            curr_fix = np.zeros((30,170))
            curr_fix = img[j:j+moving_x, k:k+moving_y].copy()


            #MD Methods
            MD_Matirx = np.zeros((30,170))
            MD_Matirx = curr_fix - atlas
            MD_Matirx = np.power(MD_Matirx, 2)
            MD = 1.0/(30.0*170.0)*np.sum(MD_Matirx)



            if(MD < MD_final):
                x_ul = j
                y_ul = k
                MD_final = MD
                print "The MD of " + str(i) + " is " + str(MD_final)



            #calculation, HMD methods
        #     atlas_greater_index = atlas > t
        #     tmp_curr_fix = np.power(( curr_fix[atlas_greater_index]*1.0 -1.0/atlas[atlas_greater_index]) , 2)
        #     curr_fix[atlas_greater_index] = tmp_curr_fix
        #     Sp_matrix = curr_fix[atlas_greater_index]
        #     # print Sp_matrix
        #     Sp_index = Sp_matrix > 0
        #     Sp = np.sum(Sp_matrix[Sp_index])
        #
        #     atlas_lower_index = atlas < t
        #     curr_fix[atlas_lower_index] = np.power(atlas[atlas_lower_index]*1.0 - 1.0*curr_fix[atlas_lower_index],2)
        #     Sn_matrix = curr_fix[atlas_lower_index]
        #     Sn = np.sum(Sn_matrix)
        #
        #
        #
        #     if Sn - Sp == 0 and Sn != 0 and Sp != 0:
        #         x_ul = j
        #         y_ul = k
        #         HDM = 0
        #         over_flag = True
        #         print "BREAK!!!"
        #         break
        #     elif np.abs(Sn - Sp) < np.abs(HDM):
        #         HDM = Sn - Sp
        #         x_ul = j
        #         y_ul = k
        #
        #         # print HDM
        # if over_flag == True:
        #     break

    # plt.imshow(img)
    # plt.show()
    print "x = " + str(x_ul)
    print "y = " + str(y_ul)
    tmp = cv2.rectangle(img, (y_ul, x_ul), (y_ul + 170, x_ul + 30),(0.8,0,0), 5)
    tmp = np.uint8(img[x_ul:x_ul+30,y_ul:y_ul+170])
    cv2.imshow("test"+str(i), img)

    cv2.waitKey(5)
cv2.waitKey()