# -*- coding: utf-8 -*-
# @Time   : 2021/4/11 21:01
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : joint_pic.py

import os

from PIL import Image


def joint_vertical(ims_list, res_name):  # 传递一个ims列表和保存名
    # 获取当前文件夹中所有JPG图像
    # im_list = [Image.open(fn) for fn in listdir() if fn.endswith('.jpg')]
    ims = [Image.open(img) for img in ims_list]

    # 图片转化为相同的尺寸
    # if RESIZE == 1:
    #
    #     height1 =
    #
    #     ims = []
    #     for i in im_list:
    #         new_img = i.resize((1480, 990), Image.BILINEAR)
    #     ims.append(new_img)

    # 单幅图像尺寸
    width, height = ims[0].size

    # 创建空白长图
    new_img = Image.new(ims[0].mode, (width, height * len(ims)))
    '''
    todolist:
    1. 对于不同尺寸的图片进行兼容,选择是否统一格式 if resize = 1
    '''

    # 拼接图片
    for i, im in enumerate(ims):
        new_img.paste(im, box=(0, i * height))

        # 保存图片
        new_img.save(res_name)


def run_joint(folders_list, res_path):
    files = os.listdir(folders_list[0])

    for file in files:
        if file.endswith('.jpg'):
            try:
                imgs = []
                # img_name = str(id_num).zfill(6)
                # '''

                for folder in folders_list:
                    pic_name = os.path.join(folder, file)
                    imgs.append(pic_name)
                '''
                folder_path1 = '../AC_OCR/Dingxiu_test/Dingxiu_2/Dingxiu_2_demo_results_res50'
                pic1 = os.path.join(folder_path1, img_name + '.jpg')
                imgs.append(pic1)
                
                folder_path2 = '../AC_OCR/Dingxiu_test/Dingxiu_2/Dingxiu_2_demo_results_newdb'
                pic2 = os.path.join(folder_path2, img_name + '.jpg')
                imgs.append(pic2)
                '''

                # res_path = '../AC_OCR/joint_compare/aug_jingbu/res18_res50_2.0_3.5'
                if not os.path.exists(res_path):
                    os.makedirs(res_path)
                joint_vertical(imgs, os.path.join(res_path, file[:-4] + '_joint.jpg'))

            except:
                print(file)


def run_joint_diff_name(folders_list, res_path):
    files = os.listdir(folders_list[0])

    for file in files:
        if file.endswith('.jpg'):
            try:
                imgs = []
                # img_name = str(id_num).zfill(6)
                # '''

                # for folder in folders_list:
                pic_name_1 = os.path.join(folders_list[0], file)
                pic_name_2 = os.path.join(folders_list[1], file)
                pic_name_3 = os.path.join(folders_list[2], file[:-4] + 'rec_uboxes_g.jpg')

                imgs.append(pic_name_1)
                imgs.append(pic_name_2)
                imgs.append(pic_name_3)
                '''
                folder_path1 = '../AC_OCR/Dingxiu_test/Dingxiu_2/Dingxiu_2_demo_results_res50'
                pic1 = os.path.join(folder_path1, img_name + '.jpg')
                imgs.append(pic1)

                folder_path2 = '../AC_OCR/Dingxiu_test/Dingxiu_2/Dingxiu_2_demo_results_newdb'
                pic2 = os.path.join(folder_path2, img_name + '.jpg')
                imgs.append(pic2)
                '''

                # res_path = '../AC_OCR/joint_compare/aug_jingbu/res18_res50_2.0_3.5'
                if not os.path.exists(res_path):
                    os.makedirs(res_path)
                joint_vertical(imgs, os.path.join(res_path, file[:-4] + '_joint.jpg'))

            except:
                print(file)


if __name__ == '__main__':
    folders_list = [
        '/Users/Beyoung/Desktop/Projects/AC_OCR/dingxiu_new/dingxiu_new_demo_results_res18_2',
        '/Users/Beyoung/Desktop/Projects/AC_OCR/dingxiu_new/dingxiu_new_demo_results_res50_2',
        '/Users/Beyoung/Desktop/Projects/AC_OCR/dingxiu_new/0A0CBAE0046F4AB7BCBFE12789547A78_res',
        # '../AC_OCR/models_test/aug_jingbu/res50_3.5',
    ]
    # run_joint(folders_list, '../AC_OCR/joint_compare/aug_jingbu/res18_res50_2.0_3.5')
    run_joint_diff_name(folders_list, '/Users/Beyoung/Desktop/Projects/AC_OCR/joint_compare/dingxiu_new/0A0CBAE0046F4AB7BCBFE12789547A78_res')
