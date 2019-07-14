# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pydicom
import numpy
import cv2

def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    img_temp = img_data.copy()
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in numpy.arange(rows):
        for j in numpy.arange(cols):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255
    return img_temp

dataset = pydicom.dcmread("../case/ser007img00007.dcm")
# dataset.pixel_array 是从零开始的灰度值，类型是numpy的二维矩阵
# dataset.image 是 CT 值，范围一般在[-1000, 1000]，类型是numpy的二维矩阵
dataset.image = dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept
print(dataset.image)
# 设置窗宽窗位是基于 CT 值进行设置
window_center = -700
window_width = 1700
lung_my = setDicomWinWidthWinCenter(dataset.image, 1500, -400, 512, 512)
lung_cv = cv2.convertScaleAbs(dataset.image-window_center, alpha=(255.0 /window_width))

print(dataset.image)
# 得到的是范围为 [0, 255] 的 numpy 二维矩阵（灰度值），即我们人可以用来观察的图像
# 另外还有 dataset.PixelData 好像存的是机器码。作用未知，我们这也没有使用
plt.subplot(121)
plt.title("my")
plt.imshow(lung_my, cmap=plt.cm.gray)
plt.subplot(122)
plt.title("cv")
plt.imshow(lung_cv, cmap=plt.cm.gray)
# 注意如果要使用 cv2 的库，部分函数要先将这个矩阵装换成 uint8 类型
# 如果没有转换下面的 imshow 的结果就会出错，某些函数也会报错
lung_cv = numpy.uint8(lung_cv)
lung_my = numpy.uint8(lung_my)
plt.show()
