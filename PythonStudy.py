import numpy as np
from PIL import Image

if __name__ == "__main__":
    threshold = 5
    img1 = Image.open('17_predict.bmp')
    img2 = Image.open('17_predict_green.bmp')
    mask_array1 = np.array(img1)
    mask_array2 = np.array(img2)
    result = np.zeros([255,255])
    # result[(mask_array1 != 0).all() and (mask_array2 != 0).all()] = 1
    for i in range(0,255):
        for j in range(0, 255):
            if mask_array1[i, j] > threshold and mask_array2[i, j] > threshold:
                if mask_array1[i, j] != 0 and  mask_array2[i, j] != 0:
                    result[i, j] = 255
    result_img = Image.fromarray(result)
    result_img.show()
    # mask_array = mask_array / 255
    # print('the shape of the mask should be {}'.format(mask_array.shape))
    # mask_convert = mask_array.copy()
    # mask_convert = 1 - mask_convert
    # mask_convert = np.stack((mask_array, mask_convert, mask_convert), axis=0)
    # print(mask_convert.shape)
    # mask = Image.fromarray(mask_convert, mode='RGB')
    print("finish")