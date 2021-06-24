# -*- coding: utf-8 -*-
# @Time   : 2021/5/28 23:50
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : jinling_test.py

import os
import time

from pdf2pic import pdf_image
from test_api_v0 import ajust_boxes


def time_change(time_init):  # 定义将秒转换为时分秒格式的函数
    time_list = []
    if time_init / 3600 > 1:
        time_h = int(time_init / 3600)
        time_m = int((time_init - time_h * 3600) / 60)
        time_s = int(time_init - time_h * 3600 - time_m * 60)
        time_list.append(str(time_h))
        time_list.append('h ')
        time_list.append(str(time_m))
        time_list.append('m ')

    elif time_init / 60 > 1:
        time_m = int(time_init / 60)
        time_s = int(time_init - time_m * 60)
        time_list.append(str(time_m))
        time_list.append('m ')
    else:
        time_s = int(time_init)

    time_list.append(str(time_s))
    time_list.append('s')
    time_str = ''.join(time_list)
    return time_str


'''
if __name__=="__main__":
    process = .0
    start = time.time()
    for i in range(total_num):

          ···
          ···
          ···

        if process < (i*1.0/total_num):
            if process != 0:
                end = time.time()
                use_time = end-start

                all_time = use_time / process
                res_time = all_time - use_time
                str_ues_time = time_change(use_time)
                str_res_time = time_change(res_time)

                print("Percentage of progress:%.0f%%   Used time:%s   Rest time:%s "%(process*100,str_ues_time,str_res_time))
            process = process + 0.01
'''


def pdf_test():
    pdf_image('/Volumes/ExtremeSSD/金陵诗徵/金陵诗徵44巻.pdf', Gray=True)
    dirpath = '/Volumes/ExtremeSSD/金陵诗徵'
    dirs = os.listdir(dirpath)

    flag = 0
    for file in dirs:
        if file.endswith('.jpg'):
            flag += 1
            print(file + ' processing: ' + str(round(flag / len(dirs) * 100, 2)) + ' %')
            ajust_boxes(os.path.join(dirpath, file))
    return


def folder_test(dirpath):
    # dirpath = '/Volumes/ExtremeSSD/金陵诗徵'
    files = os.listdir(dirpath)

    flag = 0
    # process = .01
    start = time.time()
    for file in files:
        if os.path.splitext(file)[1].lower() in IMG_EXT:
            flag += 1
            process = flag / len(files)
            # if process < (flag * 1.0 / len(files)):
            #     if process != 0:
            end = time.time()
            use_time = end - start

            rest_file = len(files) - flag
            res_time = rest_file * use_time

            # all_time = use_time / process
            # res_time = all_time - use_time
            str_ues_time = time_change(use_time)
            str_res_time = time_change(res_time)

            '''
            此处需要优化剩余时长代码,改为 记录单个文件处理时间, 乘以剩余的文件
            '''

            if os.path.exists(os.path.join(dirpath, 'output', os.path.splitext(file)[0]) + '_res_recog.txt'):
                pass
            else:
                # print(file + ' processing: ' + str(round(flag / len(files) * 100, 2)) + ' %')

                # main1(os.path.join(dirpath, file))
                ajust_boxes((os.path.join(dirpath, file)), dbg=False)
                print(file + " processing: %.0f%%   Used time:%s   Rest time:%s " % (
                    process * 100, str_ues_time, str_res_time))
                # process = process + 0.01

    return


if __name__ == '__main__':
    IMG_EXT = {'.jpg', '.png', '.tif', '.tiff', '.bmp', '.gif'}
    #
    # root_dir = '/Users/Beyoung/Desktop/Projects/AC_OCR/OCR测试图像2'
    # dirs = os.listdir(root_dir)
    # flag = 0
    # for dir in dirs:
    #     # print(dir)
    #     if os.path.isdir(os.path.join(root_dir, dir)):
    #         flag += 1
    #         print(dir, flag / len(dirs) * 100)
    #         folder_test(os.path.join(root_dir, dir))

    folder_test('/Volumes/ExtremeSSD/金陵诗徵/金陵诗徵44巻_gray')
