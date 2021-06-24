# -*- coding: utf-8 -*-
# @Time   : 2021/5/29 00:27
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : copy_file.py

import os

source_path = '/Volumes/ExtremeSSD/金陵诗徵/金陵诗徵44巻_gray/output'
target_path = '/Volumes/ExtremeSSD/金陵诗徵/金陵诗徵44巻_gray_rec'

for file in os.listdir(source_path):
    if file.endswith('res_recog.txt'):
        os.system('cp ' + os.path.join(source_path, file) + ' ' + os.path.join(target_path, file))
