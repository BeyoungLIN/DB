# -*- coding: utf-8 -*-
# @Time   : 2021/12/1 02:11
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : get_undone_folder.py

import os

def get_mini_file():
    with open('/disks/sdb/euphoria/pkg/seg_detector/logs/batch_db_predict_dingxiu.log', 'r', encoding='utf-8') as log:
        content = log.readlines()

    with open('/disks/sdb/euphoria/pkg/seg_detector/logs/batch_db_predict_dingxiu_mini.log', 'a', encoding='utf-8') as mini_log:
        for line in content[-1000:]:
            mini_log.write(line)


def get_dont_folder(root):
    folders = os.listdir(root)
    with open('done_folder.txt', 'a') as f:
        for folder in folders:
            f.write(folder + '\n')


if __name__ == '__main__':
    # with open('batch_db_pre_undone.txt', 'r', encoding='utf-8') as f:
    #     con = f.read()
    # lines = con.split(', ')
    # for line in lines:
    #     print(line.replace('\'', ''))
    #     with open('batch_db_pre_undone_list.txt', 'a', encoding='utf-8') as f_list:
    #         f_list.write(line.replace('\'', '') + '\n')
    root = '/disks/sde/euphoria/datasets/DingXiu_检查拉伸比例'
    get_dont_folder(root)