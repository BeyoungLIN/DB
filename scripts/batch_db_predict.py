# -*- coding: utf-8 -*-
# @Time   : 2021/11/27 02:05
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : batch_db_predict.py

import os



def run_predict(folder_name):
    os.system('export CUDA_VISIBLE_DEVICES=2')
    os.system(
        'python special_demo.py experiments/seg_detector/fakepages_resnet50_deform_thre.yaml --image_path /disks/sde/euphoria/datasets/DingXiu/' + folder_name + '/ --visualize --result_dir /disks/sde/euphoria/datasets/DingXiu_test/' + folder_name + '/ --resume models/fakepage_res50_iter2.bin --box_thresh 0.5 --sort_boxes')


root = '/disks/sde/euphoria/datasets/DingXiu'
folder_names = os.listdir(root)
op_root = '/disks/sde/euphoria/datasets/DingXiu_检查拉伸比例'  # 输出可视化图片可以检查

for folder_name in folder_names:
    folder_pth = os.path.join(root, folder_name)  # 文件路径
    if os.path.isdir(folder_pth):
        print(folder_pth)
        files = os.listdir(folder_pth)
        # folder_name = folder.split('/')[-1]  # 获得原文件夹名
        # print(folder)  # /disks/sdc/euphoria/datasets/DingXiu/0AFAEE0760574746B10E6A80ADA611B1
        fd_op_pth = os.path.join(op_root, folder_name)
        print('fd_op_pth', fd_op_pth)
        if not os.path.exists(fd_op_pth):
            os.mkdir(fd_op_pth)
        for file in files:
            if os.path.splitext(file)[1].lower() in IMG_EXT:  # 读取所有图片文件
                # try:
                #  读res
                # 保存原名txt,txt-size
                # 可视化原图
                file_path = os.path.join(folder, file)  # 原文件夹内图片路径
                txt_path = os.path.join(folder, 'res_' + file.split('.')[0] + '.txt')  # 图片对应的txt名
                pic_boxes = get_pure_box(txt_path)
                pic_boxes = lengthen_box(pic_boxes, 0.011)
                # boxes = get_singel_line_box(boxes, mod='cv')
                cv_draw_box(file_path, pic_boxes, op_pth=fd_op_pth, color='b')  # 蓝色线

