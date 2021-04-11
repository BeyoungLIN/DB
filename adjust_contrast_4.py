import cv2
import numpy as np


def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img1.shape

    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    # cv2.imshow('original_img', img)
    cv2.imshow("contrast_img", dst)
    cv2.imwrite('/Users/Beyoung/Desktop/Projects/AC_OCR/3.22竞赛数据/imgs/image_032_contrast.tif', dst)


img = cv2.imread("/Users/Beyoung/Desktop/Projects/AC_OCR/3.22竞赛数据/imgs/image_032.tif", cv2.IMREAD_COLOR)
contrast_img(img, 1.3, 10)
cv2.waitKey(0)
cv2.destroyAllWindows()

