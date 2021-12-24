# -*- coding: utf-8 -*-
# @Time   : 2021/11/27 02:05
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : batch_db_predict.py

import os
import cv2
import argparse

import time

import math
import torch
import numpy as np
import itertools

from experiment import Structure, Experiment
from concern.config import Configurable, Config
import utils

from special_demo import Demo

# CUDA_VISIBLE_DEVICES = 1
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def run_db_model(folder_ip_pth_, folder_op_pth_):
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    # parser.add_argument('exp', default='experiments/seg_detector/fakepages_resnet50_deform_thre.yaml', type=str)
    # parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    # parser.add_argument('--image_path', type=str, help='image path')
    # parser.add_argument('--data', type=str,
    #                     help='The name of dataloader which will be evaluated on.')
    # parser.add_argument('--image_short_side', type=int, default=736,
    #                     help='The threshold to replace it in the representers')
    # parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    # parser.add_argument('--thresh', type=float,
    #                     help='The threshold to replace it in the representers')
    # parser.add_argument('--box_thresh', type=float, default=0.6,
    #                     help='The threshold to replace it in the representers')
    # parser.add_argument('--visualize', action='store_true',
    #                     help='visualize maps in tensorboard', default=True)
    # parser.add_argument('--resize', action='store_true',
    #                     help='resize')
    # parser.add_argument('--polygon', action='store_true',
    #                     help='output polygons if true')
    # parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
    #                     help='Show images eagerly')
    # parser.add_argument('--sort_boxes', action='store_true', dest='sort_boxes',
    #                     help='Sort boxes for further works', default=True)

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    args['exp'] = 'experiments/seg_detector/fakepages_resnet50_deform_thre.yaml'
    args['image_path'] = folder_ip_pth_
    args['result_dir'] = folder_op_pth_
    args['resume'] = 'models/fakepage_res50_iter2.bin'
    args['visualize'] = 1
    args['sort_boxes'] = 0.6
    args['sort_boxes'] = 1

    # '--image_path ' + folder_ip_pth_ + '/ \\\ \\\n--result_dir ' + folder_op_pth_ + '/ \\\n--resume \\\nmodels/fakepage_res50_iter2.bin\\\n'

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    # Delete train settings, prevent of reading training dataset
    experiment_args.pop('train')
    experiment_args.pop('evaluation')
    experiment_args.pop('validation')
    experiment = Configurable.construct_class_from_config(experiment_args)

    demo_handler = Demo(experiment, experiment_args, cmd=args)
    error_pic = []

    if os.path.isdir(args['image_path']):
        img_cnt = len(os.listdir(args['image_path']))
        for idx, img in enumerate(os.listdir(args['image_path'])):
            if os.path.splitext(img)[1].lower() not in ['.jpg', '.tif', '.png', '.jpeg', '.gif']:
                continue
            t = time.time()
            try:
                demo_handler.inference(os.path.join(args['image_path'], img), args['visualize'])
            except:
                error_pic.append(img)

            print("{}/{} elapsed time : {:.4f}s".format(idx + 1, img_cnt, time.time() - t))
    else:
        t = time.time()
        demo_handler.inference(args['image_path'], args['visualize'])
        print("elapsed time : {}s".format(time.time() - t))
    print(error_pic)


def get_pure_box(file_pth):
    '''
    输入在只有八点坐标 + size 的txt
    1416,36,1461,37,1447,579,1402,578,small
    需要做一些变换,只取前八点,转为int,注意末尾有\n
    :return: 二维数组 [[1416, 36, 1461, 37, 1447, 579, 1402, 578], [737, 272, 776, 275, 770, 388, 730, 386]]
    '''
    with open(file_pth, 'r', encoding='utf-8') as ip_file:
        boxes = ip_file.readlines()
    new_boxes = []
    for box in boxes:
        box = box.replace('\n', '')
        box = box.split(',')[:8]  # 下标8不取值
        box = [int(bo) for bo in box]  # 此处转为int
        new_boxes.append(box)
    # print(new_boxes)
    return new_boxes


def get_file_box(file_pth):
    boxes = []
    with open(file_pth, 'r') as bf:
        lines = bf.readlines()
        bf.close()
    for line in lines:
        boxes_temp = line.replace('\n', '')
        boxes_temp = boxes_temp.split(',')
        for box in boxes_temp:
            boxes.append(int(box))
    # print(boxes)
    return boxes


def lengthen_box(boxes_list, ratio):
    '''
    :param boxes_list:
    :param ratio:
    :return: 延长某几条线的new_box
    '''
    new_box = []
    for box in boxes_list:
        box[5] = int(box[5] * (1 + ratio))
        box[7] = int(box[7] * (1 + ratio))
        new_box.append(box)
    # print(new_box)
    return new_box


def sav_box_txt(sav_box, res_txt_pth='', txt_sav_pth='', txt_size_pth='', sav_size=1):
    box_content = ''
    box_content_size = ''
    with open(res_txt_pth, 'r', encoding='utf=8') as res_f:
        lines = res_f.readlines()
    sizes = [line.split(',')[-1].replace('\n', '') for line in lines]
    i = 0
    for box in sav_box:
        for cor in box:
            box_content += (str(cor) + ',')
            box_content_size += (str(cor) + ',')
        box_content_size = box_content_size + str(sizes[i]) + '\n'
        box_content = box_content[:-1] + '\n'
        # box_content += '\n'
        i += 1
    # print(box_content)
    # print(box_content_size)
    with open(txt_sav_pth, 'w', encoding='utf-8') as txt_f:
        txt_f.write(box_content)
    if sav_size:
        with open(txt_size_pth, 'w', encoding='utf-8') as txt_size_f:
            txt_size_f.write(box_content_size)


def cv_draw_box(ip_pic_path, boxes, op_pth='', mod='', color='red'):
    image = cv2.imread(ip_pic_path)
    # GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # h, w = image.shape[:2]
    # h, w = map(int, [h/4, w/4])
    # print(h,w)
    # # no flip
    # draw_0 = cv2.rectangle(image, (100, 100), (10, 10), (0, 0, 255))#cv2.rectangle(image, pt1,pt2, color)
    for box in boxes:
        for x1 in range(0, len(box) - 1, 2):
            y1 = x1 + 1
            if x1 == (len(box) - 2):
                # print(0)
                x2 = 0
                y2 = 1
            else:
                x2 = x1 + 2
                y2 = x1 + 3
            # x_0, y_0, x_1, y_1, x_2, y_2 = box[], box[1], box[2], box[3], box[4]
            if color == 'b':
                image = cv2.line(image, (box[x1], box[y1]), (box[x2], box[y2]), (255, 0, 0), 2)
            elif color == 'g':
                image = cv2.line(image, (box[x1], box[y1]), (box[x2], box[y2]), (0, 255, 0), 1)
            else:
                image = cv2.line(image, (box[x1], box[y1]), (box[x2], box[y2]), (0, 0, 255), 3)  # 红色

    # x, y, w, h = cv2.boundingRect(GrayImage)
    # 参数：pt1,对角坐标１, pt2:对角坐标２
    # 注意这里根据两个点pt1,pt2,确定了对角线的位置，进而确定了矩形的位置
    # The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners are pt1 and pt2.
    # draw_0 = cv2.rectangle(image, (2 * w, 2 * h), (3 * w, 3 * h))

    # cv2.imwrite("vertical_flip.jpg", draw_1)  # 将画过矩形框的图片保存到当前文件夹

    # cv2.imshow("image", image)  # 显示画过矩形框的图片
    # cv2.waitKey(0)
    # cv2.destroyWindow("image")
    # print('op_pth', op_pth)
    save_pth = os.path.join(op_pth, ip_pic_path.split('/')[-1].split('.')[0] + '.jpg')
    # print('save_pth', save_pth)
    cv2.imwrite(save_pth, image)
    return save_pth


def run_predict(folder_ip_pth_, folder_op_pth_):
    os.system('export CUDA_VISIBLE_DEVICES=2')
    os.system(
        'python special_demo.py experiments/seg_detector/fakepages_resnet50_deform_thre.yaml --image_path ' + folder_ip_pth_ + '/ --visualize --result_dir ' + folder_op_pth_ + '/ --resume models/fakepage_res50_iter2.bin --box_thresh 0.5 --sort_boxes')


def run_predict_sh(folder_ip_pth_, folder_op_pth_):
    content = '\nexport CUDA_VISIBLE_DEVICES=1\n\npython special_demo.py \\\nexperiments/seg_detector/fakepages_resnet50_deform_thre.yaml \\\n--image_path ' + folder_ip_pth_ + '/ \\\n--visualize \\\n--result_dir ' + folder_op_pth_ + '/ \\\n--resume \\\nmodels/fakepage_res50_iter2.bin \\\n--box_thresh 0.5 \\\n--sort_boxes \\\n\n'
    with open('scripts/run_special_demo_db2.0.sh', 'w', encoding='utf-8') as sh_f:
        sh_f.write(content)
    os.system('sh scripts/run_special_demo_db2.0.sh')


def read_todo_file(todo_pth):
    todo_list = [s.rstrip() for s in open(todo_pth, 'r', encoding='utf-8').readlines()]
    return todo_list


def get_db_res(ori_folder, res_folder):
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    # 对该文件夹做预测
    run_predict_sh(ori_folder, res_folder)
    # run_db_model(ori_folder, res_folder)


def lengthen_res_visualize(vs_root, dir_name, pic_pth, boxes):
    folder_visualize = os.path.join(vs_root, dir_name)
    if not os.path.exists(folder_visualize):
        os.mkdir(folder_visualize)
    # boxes = get_singel_line_box(boxes, mod='cv')
    cv_draw_box(pic_pth, boxes, op_pth=folder_visualize, color='b')  # 蓝色线为延长后的线


def save_ckpt(tmp_list, saved_list, op_txt_pth_='', limit_num=10):
    if len(tmp_list) > limit_num:
        with open(op_txt_pth_, 'a', encoding='utf-8') as update_todo_f:
            for todo_f in tmp_list:
                update_todo_f.write(todo_f + '\n')
        # if if_combine:
        #     with open(done_file_ori, 'a', encoding='utf-8') as update_todo_f_ori:
        #         for todo_f in done_file_tmp:
        #             update_todo_f_ori.write(todo_f + '\n')
        saved_list += tmp_list
        tmp_list = []
    return tmp_list, saved_list


def init_read_ckpt(ckpt_pth, combine=1, new_ckpt_pth=''):
    '''
    :param ckpt_pth:传入一个txt, 可以把里面的文件夹或文件转为列表
    :param combine: 1 为覆盖ckpt, 2 为存为两个文件, 即有一个输出文件, 0 为不保存新增文件(此选项未完成相关代码)
    :param new_ckpt_pth: combine为2时的输出路径
    :return:
    '''
    if not os.path.exists(ckpt_pth):
        blank_txt = open(ckpt_pth, 'w', encoding='utf-8')
    if combine == 1:
        new_ckpt_pth = ckpt_pth
    elif combine == 2:
        new_ckpt_pth = ckpt_pth[:-4] + '_copy.txt'
        os.system('cp ' + ckpt_pth + ' ' + new_ckpt_pth)
        new_ckpt_pth = new_ckpt_pth
    elif combine == 0:
        new_ckpt_pth = ''
    ckpt2list = read_todo_file(ckpt_pth)
    tmp = []
    return tmp, ckpt2list, new_ckpt_pth


def read_ckpt_list(ckpt_txt_list, mod='update', new_ckpt_pth=''):
    ckpt_list = []
    if not os.path.exists(ckpt_txt_list[-1]):
        with open(new_ckpt_pth, 'w', encoding='utf-8') as blank_txt:
            blank_txt.close()
    # if mod == 'update':
    #     new_ckpt_pth = ckpt_txt_list[-1]
    # elif mod == 'newbatch':
    #     if new_ckpt_pth == '':
    #         new_ckpt_pth = ckpt_txt_list[-1]
    #     with open(new_ckpt_pth, 'w', encoding='utf-8') as blank_txt:
    #         blank_txt.close()
    new_ckpt_pth = ckpt_txt_list[-1]
    for one_txt in ckpt_txt_list:
        one_txt_done_list = read_todo_file(one_txt)
        ckpt_list += one_txt_done_list
    tmp = []
    print(tmp[:-50] if len(tmp) > 51 else tmp)

    return tmp, ckpt_list, new_ckpt_pth


IMG_EXT = {'.jpg', '.png', '.tif', '.tiff', '.bmp', '.gif'}
root = '/disks/sdg/euphoria/datasets/DingXiu/'
db_res_root = '/disks/sdg/euphoria/datasets/DingXiu_test/'
visualize_root = '/disks/sdg/euphoria/datasets/DingXiu_检查拉伸比例'  # 输出可视化图片可以检查
# 初始化并读取已完成的文件列表
done_dir_txt_list = [
    # '/disks/sdb/euphoria/pkg/seg_detector/extract/done_folder_1.txt',  # 第一批 x-12.1 2000
    # '/disks/sdb/euphoria/pkg/seg_detector/extract/done_folder_2.txt',  # 第二批 12.2-
    # '/disks/sdb/euphoria/pkg/seg_detector/extract/done_folder_3.txt',  # 第三批 12.20-1222
    '/disks/sdb/euphoria/pkg/seg_detector/extract/done_folder_1222_1.txt',  # 第一批 12.22
]
# finished_folder_tmp, finished_folder_list, new_finished_folder_pth = \
    # read_ckpt_list(done_dir_txt_list, 'update')
finished_folder_tmp, finished_folder_list, new_finished_folder_pth = \
    read_ckpt_list(done_dir_txt_list)

done_file_txt_list = [
    # '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_1.txt',  # 第一批 x-12.1 60w
    # '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_2.txt',  # 第二批 12.1-
    # '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_3.txt',  # 第三批 12.20-
    '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_1222_1.txt',  # 第一批 12.22
]
# done_file_tmp, done_file_list, new_done_file_pth = read_ckpt_list(done_file_txt_list, 'update')
done_file_tmp, done_file_list, new_done_file_pth = read_ckpt_list(done_file_txt_list)

# finished_folder_pth = '/disks/sdb/euphoria/pkg/seg_detector/extract/done_folder_1201_combine.txt'
# finished_folder_tmp, finished_folder_list, new_finished_folder_pth = init_read_ckpt(finished_folder_pth, combine=1)
# done_file_pth = '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_backup.txt'
# done_file_tmp, done_file_list, new_done_file_pth = init_read_ckpt(done_file_pth, combine=1)
no_txt = []
count = 0
no_txt_flag = 0
folder_names = os.listdir(root)  # DingXiu里面的所有文件名
folder_names_copy = folder_names
for folder_name in folder_names:
    if folder_name not in (finished_folder_list + finished_folder_tmp):  # 不在两个列表内, 目前只收集文件名
        folder_pth = os.path.join(root, folder_name)  # 原文件夹路径
        folder_db_res = os.path.join(db_res_root, folder_name)
        if os.path.isdir(folder_pth):
            print(folder_pth)
            get_db_res(folder_pth, folder_db_res)
            # 读取原鼎秀文件夹, 获取所有文件名
            files = os.listdir(folder_pth)  # 此时的file后缀为.png
            for file in files:
                # 读取所有图片文件
                if os.path.splitext(file)[1].lower() in IMG_EXT:
                    # test文件夹内图片路径
                    db_res_pic_pth = os.path.join(folder_db_res, file.split('.')[0] + '.jpg')
                    res_txt_pth = os.path.join(folder_db_res, 'res_' + file.split('.')[0] + '.txt')  # 文件夹名为DingXiu_test
                    new_txt_pth = os.path.join(folder_db_res, file.split('.')[0] + '.txt')
                    new_txt_size_pth = os.path.join(folder_db_res, file.split('.')[0] + '_size.txt')
                    # 得到变换的坐标
                    if not os.path.exists(res_txt_pth):
                        res_txt_pth = new_txt_pth.replace('DingXiu_test', 'DingXiu')  # 不存在的情况下使用旧坐标
                        db_res_pic_pth = os.path.join(folder_pth, file)
                        if not os.path.exists(res_txt_pth):
                            no_txt.append(res_txt_pth)
                            no_txt_flag = 1

                    if no_txt_flag == 0:
                        pic_boxes = get_pure_box(res_txt_pth)
                        pic_boxes = lengthen_box(pic_boxes, 0.02)
                        # 保存原名txt,txt-size
                        sav_box_txt(pic_boxes, res_txt_pth=res_txt_pth, txt_sav_pth=new_txt_pth, txt_size_pth=new_txt_size_pth)

                        # 将txt文件夹复制到dingxiu文件夹中
                        common = 'cp ' + new_txt_pth + ' ' + new_txt_pth.replace('DingXiu_test', 'DingXiu')
                        os.system(common)

                        # 保留DingXiu路径的done_file 文件级png的append,用于后续creat_true_data
                        done_file_tmp.append(db_res_pic_pth.replace('DingXiu_test', 'DingXiu').replace('jpg', 'png'))  # 转为DingXiu路径, png格式
                        # 文件级的写入
                        done_file_tmp, done_file_list = save_ckpt(tmp_list=done_file_tmp, saved_list=done_file_list, limit_num=100, op_txt_pth_=new_done_file_pth)  # tmp在前, 完整list在后
                        count += 1
                        if count % 1000 == 0:
                            print('stage_count', count)
                            print(db_res_pic_pth)

                            # 可视化原图   此处设计为每1000张检查一次
                            # if not os.path.exists(db_res_pic_pth):
                            #     db_res_pic_pth =
                            lengthen_res_visualize(visualize_root, folder_name, db_res_pic_pth, pic_boxes)
                    else:
                        no_txt_flag = 0
                        print(no_txt)

                # folder_pth 是文件夹才能做判断
            finished_folder_tmp.append(folder_name)
            finished_folder_tmp, finished_folder_list = save_ckpt(tmp_list=finished_folder_tmp, saved_list=finished_folder_list, limit_num=10, op_txt_pth_=new_finished_folder_pth)
            # except Exception as e:
            #     print('错误类型是', e.__class__.__name__)
            #     print('错误明细是', e)
        print('count', count)
        folder_names_copy.remove(folder_name)
        print('unfinished_folder_num:', len(folder_names_copy))

