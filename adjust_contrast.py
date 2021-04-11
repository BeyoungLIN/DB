# -*- encoding: utf-8 -*-
__author__ = 'beyoung'

import cv2

img = cv2.imread('/Users/Beyoung/Desktop/Projects/AC_OCR/3.22竞赛数据/imgs/image_032.tif')
cv2.imshow('original_img', img)
rows, cols, channels = img.shape
dst = img.copy()

a = 0.5
b = 80
for i in range(rows):
    for j in range(cols):
        for c in range(3):
            color = img[i, j][c] * a + b
            if color > 255:  # 防止像素值越界（0~255）
                dst[i, j][c] = 255
            elif color < 0:  # 防止像素值越界（0~255）
                dst[i, j][c] = 0

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()