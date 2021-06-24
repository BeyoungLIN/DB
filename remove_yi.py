# -*- coding: utf-8 -*-
# @Time   : 2021/5/30 14:21
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : remove_yi.py

import os


def wash_redun(source_path, tar_path):
    with open(source_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            # print(line)
            # if not len(line) == 1:
            if not ((line.startswith('一') or line.startswith('二') or line.startswith('也')) and (
                    len(line) == 2 or len(line) == 1)):
                # if line.endswith('一\n'):
                #     print(line[:-2])
                #     with open(tar_path, 'a') as of:
                #         of.write(line[:-2])
                # else:
                print(line)
                with open(tar_path, 'a') as of:
                    of.write(line)

    return


# input_path = '/Volumes/ExtremeSSD/金陵诗徵/金陵诗徵44巻_gray/output/金陵诗徵44巻_4149_res_recog.txt'
# output_path = input_path + 'washed'

root_path = 'AC_OCR/金陵诗徵44巻_recog_res'
output_path = root_path + '_washed'
if not os.path.exists(output_path):
    os.mkdir(output_path)

for file in os.listdir(root_path):
    if file.endswith('.txt'):
        wash_redun(os.path.join(root_path, file), os.path.join(output_path, file))
#         # os.system('cp ' + os.path.join(root_path, file) + ' ' + os.path.join(target_path, file))
