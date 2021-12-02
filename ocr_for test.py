# -*- coding: utf-8 -*-
# @Time   : 2021/6/17 01:00
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : get_size_pic.py

import os
import time

# from copy_files_4ocr import *
from test_api_v0 import ajust_boxes, test_one_adv
from copy_files_4ocr import copy_file

def get_double_folder_linesize(root_path):
    # root_path = '/Users/Beyoung/Desktop/Projects/AC_OCR/OCR测试图像2'
    error_pics = []
    dirs = os.listdir(root_path)
    for dire in dirs:
        dir_path = os.path.join(root_path, dire)
        # print(dir_path)
        if os.path.isdir(dir_path):
            dir_path_2 = os.path.join(dir_path)
            files = os.listdir(dir_path_2)
            for file in files:
                if os.path.splitext(file)[1].lower() in IMG_EXT:
                    file_path = os.path.join(dir_path_2, file)
                    try:
                        ajust_boxes(file_path, dbg=False)
                        test_one_adv(file_path, mod='adv')
                        test_one_adv(file_path, mod='mix')
                    except:
                        error_pics.append(file)
            print(error_pics)


def get_single_folder_linesize(root_dir):
    # root_dir = '/disks/sde/beyoung/files_processor/宝庆'
    files = os.listdir(root_dir)
    error = []
    flag = 0
    for file in files:
        if os.path.splitext(file)[1].lower() in IMG_EXT:
            file_path = os.path.join(root_dir, file)
            # try:
            ajust_boxes(file_path, dbg=False)
            # test_one_adv(file_path, mod='adv')
            test_one_adv(file_path, mod='mix')
            flag += 1
                # read_json_2txt(file_path[:-4] + '_resapi_mix.json.txt')
            # except:
            #     error.append(file)
    print('处理文件数:', flag)
    print(error)


def process(root_list, mod='single'):
    for root in root_list:
        start_time = time.time()
        if mod == 'single':
            get_single_folder_linesize(root)
        if mod == 'double':
            get_double_folder_linesize(root)
        # sin_folders(root)
        source_path = os.path.join(root, 'output')
        files = os.listdir(source_path)
        target_path = root + '_res'
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for file in files:
            if os.path.isfile(os.path.join(source_path, file)):
                os.system('cp ' + os.path.join(source_path, file) + ' ' + os.path.join(target_path, file))
        # copy_file(root)
        end_time = time.time()
        used_time = end_time - start_time
        print(root + '\n处理时间', used_time)


if __name__ == '__main__':
    IMG_EXT = {'.jpg', '.png', '.tif', '.tiff', '.bmp', '.gif'}
    # root_path = '/disks/sde/beyoung/files_processor/OCR测试图像2'

    # root_path = '/disks/sdd/beyoung/data/ER007'
    # root_path = '/disks/sdd/beyoung/data/测试7.5'
    root_list = [
        # '/disks/sdd/beyoung/data/國家圖書館藏敦煌遺書_001',
        # '/disks/sdd/beyoung/data/2563[函368]',
        # '/disks/sdd/beyoung/data/纂図互註荀子3',
        # '/disks/sde/beyoung/files_processor/6059.桐南凤岗李氏宗谱：三十二卷：[桐庐]',
        # '/disks/sdd/beyoung/data/pkuocrtest-20210705',
        # '/disks/sdd/beyoung/data/經問卷一',
        # '/disks/sdb/euphoria/DB/datasets/ER007/ER007_jpg',
        '/Users/Beyoung/Desktop/Projects/corpus/00025的副本1500',

    ]

    # process(root_list, 'double')
    # process(root_list, 'single')
    # copy_file(root_list, 'single')
    # copy_file(root_list, 'double')

    # get_double_folder_linesize(root_path)
    # single_file = '/disks/sdd/beyoung/data/ZHSY000116-000009_gray.png'
    # single_file = '/disks/sdd/beyoung/data/ZHSY000116-000009_rem_red.png'
    # single_file = '/disks/sdd/beyoung/data/error/IMG_20210414_141531.jpg'
    # single_file = '/disks/sdb/euphoria/DB/datasets/ER007/001029.jpg'
    # single_file = '/disks/sdb/euphoria/DB/datasets/ER007/001029.jpg'
    # single_file = '/Users/Beyoung/Desktop/Projects/corpus/00025的副本1500/ZHSY000025-000296.tif'
    single_file_list = [
        # '/Users/Beyoung/Desktop/Projects/corpus/00025的副本1500/ZHSY000025-000296.tif',
        # '/Users/Beyoung/Desktop/Projects/corpus/00025的副本1500/ZHSY000025-000296的副本.tif',
        # '/Users/Beyoung/Desktop/Projects/corpus/ZHSY000025-000296fortest/ZHSY000025-000296的副本_contrast.tif',
        # '/Users/Beyoung/Desktop/Projects/corpus/ZHSY000025-000296fortest/ZHSY000025-000296_contrast.tif',
        '/Users/Beyoung/Desktop/Projects/corpus/ZHSY000025-000296fortest/ZHSY000025-000296.tif',
        '/Users/Beyoung/Desktop/Projects/corpus/ZHSY000025-000296fortest/ZHSY000025-000269.tif',
        '/Users/Beyoung/Desktop/Projects/corpus/ZHSY000025-000296fortest/ZHSY000025-000265.tif',
        '/Users/Beyoung/Desktop/Projects/corpus/ZHSY000025-000296fortest/ZHSY000025-000261.tif',
        '/Users/Beyoung/Desktop/Projects/corpus/ZHSY000025-000296fortest/ZHSY000025-000258.tif',
        '/Users/Beyoung/Desktop/Projects/corpus/ZHSY000025-000296fortest/ZHSY000025-000243.tif',
    ]
    for single_file in single_file_list:
        ajust_boxes(single_file, dbg=False)
        # test_one_adv(single_file, mod='adv')
        # test_one_adv(single_file, mod='mix')
