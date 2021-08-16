# -*- coding: utf-8 -*-
# @Time   : 2021/8/15 22:54
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : k_means.py.py

import random
import time

import numpy as np


def func00():  # 生成随机数列表

    # random.seed(1)
    kjz1 = [random.randint(1, 50) for j in range(0, 7000)]
    kjz1.extend([random.randint(80, 150) for j in range(0, 8000)])
    kjz1.extend([random.randint(200, 300) for j in range(0, 5000)])
    kjz1.extend([random.randint(400, 500) for j in range(0, 8000)])
    return kjz1


def func01(kjz1):  # 2分类

    bj = 1
    kjz1 = np.sort(kjz1)
    while (True):
        if bj == 1:
            kj = np.mean([kjz1[0], kjz1[len(kjz1) - 1]])  # 初始分组均值使用最小值和最大值的平均值
        else:
            k1 = s1
            k2 = s2
            kj = np.mean([k1, k2])
        kjz2 = [[], []]
        for j in kjz1:
            if j <= kj:
                kjz2[0].append(j)
            else:
                kjz2[1].append(j)
        s1 = np.mean(kjz2[0])
        s2 = np.mean(kjz2[1])
        if bj == 2:
            if s1 == k1 and s2 == k2:
                break
        bj = 2
    return kjz2


def func02(kjz1, k):  # k个均值分k份

    kjz1 = np.sort(kjz1)  # 正序
    wb2 = kjz1.copy()
    # 初始均匀分组wb1
    xlb = []
    a = round(len(wb2) / (k))
    b = len(wb2) % (k)
    for j in range(1, k + 1):
        xlb.append(j * a)
        if j == k:
            xlb[j - 1] = xlb[j - 1] + b
    j = 0
    wb1 = []
    for j in range(0, k):
        wb1.append([])
    i = 0
    j = 0
    while (i <= len(wb2) - 1):
        wb1[j].append(wb2[i])
        if i >= xlb[j] - 1:
            j = j + 1
        i = i + 1
    kj1 = means(wb1)  # 初始分组均值

    bj = 1
    while (True):
        wb2 = kjz1.copy().tolist()
        if bj != 1:
            kj1 = kj2.copy()

        wb3 = []
        for j in range(0, k - 1):
            wb3.append([])
        for j in range(0, k - 1):
            i = -1
            while (True):
                if wb2[i] <= kj1[j]:
                    wb3[j].append(wb2.pop(i))
                else:
                    i = i + 1
                if i >= len(wb2):
                    break
        wb3.append(wb2)

        kj2 = means(wb3)  # 过程均值
        if bj == 2:
            if kj1 == kj2:
                break
        bj = 2
    return wb3


def means(lb1):  # 计算均值

    mean1 = []
    mean2 = []
    std1 = []
    for j in lb1:
        mean1.append(np.mean(j).tolist())
    for j in range(1, len(mean1)):
        mean2.append(np.mean([mean1[j - 1], mean1[j]]))  # 分组均值使用各组的均值
    print(mean2)
    return mean2


if __name__ == '__main__':

    start = time.time()

    # kjz1 = func00()  # 生成随机数列表
    list = [[1142.0, 27.0, 1232.0, 26.0, 1242.0, 585.0, 1152.0, 586.0], [986.0, 26.0, 1044.0, 26.0, 1044.0, 189.0, 986.0, 189.0], [890.0, 115.0, 977.0, 113.0, 983.0, 404.0, 897.0, 406.0], [833.0, 671.0, 903.0, 671.0, 903.0, 753.0, 833.0, 753.0], [769.0, 203.0, 873.0, 203.0, 877.0, 671.0, 773.0, 672.0], [771.0, 673.0, 843.0, 673.0, 843.0, 752.0, 771.0, 752.0], [715.0, 759.0, 782.0, 759.0, 782.0, 839.0, 715.0, 839.0], [658.0, 205.0, 755.0, 204.0, 761.0, 753.0, 664.0, 754.0], [652.0, 760.0, 722.0, 760.0, 722.0, 843.0, 652.0, 843.0], [595.0, 670.0, 660.0, 670.0, 660.0, 752.0, 595.0, 752.0], [544.0, 207.0, 643.0, 206.0, 646.0, 664.0, 548.0, 665.0], [534.0, 673.0, 600.0, 673.0, 600.0, 757.0, 534.0, 757.0], [417.0, 202.0, 532.0, 201.0, 536.0, 580.0, 421.0, 581.0], [475.0, 579.0, 542.0, 579.0, 542.0, 657.0, 475.0, 657.0], [414.0, 579.0, 482.0, 579.0, 482.0, 657.0, 414.0, 657.0], [299.0, 199.0, 410.0, 199.0, 410.0, 582.0, 299.0, 582.0], [354.0, 580.0, 421.0, 580.0, 421.0, 659.0, 354.0, 659.0], [294.0, 579.0, 363.0, 579.0, 363.0, 659.0, 294.0, 659.0], [236.0, 762.0, 306.0, 762.0, 306.0, 843.0, 236.0, 843.0], [185.0, 201.0, 287.0, 201.0, 287.0, 753.0, 185.0, 753.0], [174.0, 764.0, 245.0, 764.0, 245.0, 844.0, 174.0, 844.0], [70.0, 205.0, 173.0, 205.0, 173.0, 750.0, 70.0, 750.0], [113.0, 762.0, 180.0, 762.0, 180.0, 843.0, 113.0, 843.0], [51.0, 764.0, 122.0, 764.0, 122.0, 843.0, 51.0, 843.0]]

    kjz1 = []

    for box in list:
        width = box[2] - box[0]
        kjz1.append(width)
    print(kjz1)

    # kjz2 = func01(kjz1)  # 2分类
    # for j in kjz2:
    #     print(len(j))

    k = 3
    kjz3 = func02(kjz1, k)  # k个均值分k份
    print(kjz3)
    for j in kjz3:
        print(len(j))

    print('Time used:', int((time.time() - start) / 60 * 10) / 10, '分钟')
