# -*- coding: utf-8 -*-
# @Time   : 2021/5/28 23:50
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : jinling_test.py

import os

from pdf2pic import pdf_image
from test_api_v0 import main1

# pdf_image('/Volumes/ExtremeSSD/金陵诗徵/金陵诗徵44巻.pdf')

dirpath = '/Volumes/ExtremeSSD/金陵诗徵'
dirs = os.listdir(dirpath)

flag = 0
for file in dirs:
    if file.endswith('.jpg'):
        flag += 1
        print(file + ' processing: ' + str(round(flag / len(dirs) * 100, 2)) + ' %')
        main1(os.path.join(dirpath, file))
