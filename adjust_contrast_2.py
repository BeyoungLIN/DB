#!usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------
"""
@author:Sui yue
@describe: 对比增强，线性变换
@time: 2019/09/15 14:21:44
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


# 主函数

def calcGrayHist(image):
    # 灰度图像矩阵的高、宽
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
            # 显示灰度直方图
    # 画出灰度直方图
    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=2, c='black')
    # 设置坐标轴的范围
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue])
    plt.ylabel('gray level')
    plt.ylabel("number or pixels")
    # 显示灰度直方图
    plt.show()


if __name__ == "__main__":
    # 读图像
    I = cv2.imread('/Users/Beyoung/Desktop/Projects/AC_OCR/3.22竞赛数据/imgs/image_032.tif', cv2.IMREAD_GRAYSCALE)
    # 线性变换
    a = 3
    O = float(a) * I
    # 进行数据截断，大于255 的值要截断为255
    O[0 > 255] = 255
    # 数据类型转换
    O = np.round(O)
    # uint8类型
    O = O.astype(np.uint8)
    # 显示原图和线性变换后的效果
    cv2.imshow("I", I)
    cv2.imshow("O", O)
    calcGrayHist(I)
    calcGrayHist(O)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
