# -*- coding: utf-8 -*-
# @Time   : 2021/8/15 11:32
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : craft_char.py.py
import base64
import collections
import itertools
import json
import math
import os
import traceback

import cv2
import numpy as np
import requests
import six
from PIL import Image
from sklearn.cluster import DBSCAN, MeanShift, OPTICS, Birch

from pic_projection_2 import box_cut_vertical
from postdetect import filter_subbox, randcolors, draw_line, line_interact, conv_cords, get_w_rngs, get_line_size, \
    union_subboxes, \
    ygroup_uboxes, re_mapping_lsize
from zhtools.langconv import *

# from skimage import io

url_line_detect = 'http://api.chinesenlp.com:7001/ocr/v1/line_detect'
url_line_recog = 'http://api.chinesenlp.com:7001/ocr/v1/line_recog'

url_page_recog_0 = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog_0'  # craft_char
url_page_recog = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog'  # craft行
url_page_recog_1 = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog_1'  # db


def cv_imread(file_path=""):
    img_mat = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img_mat


def cv_imwrite(file_path, frame):
    cv2.imencode('.jpg', frame)[1].tofile(file_path)


def base64of_img(pth_img):
    image_base64 = ''
    with open(pth_img, 'rb') as f:
        image = f.read()
        image_base64 = str(base64.b64encode(image), encoding='utf-8')
    return image_base64


def request_api(url_api, params, _auth):
    r = requests.post(url_api, data=params, auth=_auth)
    str_res = r.text
    o_res = json.loads(str_res)
    res = o_res['data']
    return res


def readPILImg(pth_img):
    img_base64 = base64of_img(pth_img)
    img = base64_to_PIL(img_base64)
    return img


def base64_to_PIL(string):
    """
    base64 string to PIL
    """
    try:
        base64_data = base64.b64decode(string)
        buf = six.BytesIO()
        buf.write(base64_data)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
    except Exception as e:
        print(e)
        return None


def scale_x(_cords, resize_x):
    xmin, ymin, xmax, ymax = _cords
    xmid = (xmin + xmax) / 2
    xmin_ = math.ceil(xmid - resize_x * (xmid - xmin))
    xmax_ = math.floor(xmid + resize_x * (xmax - xmid))

    return xmin_, ymin, xmax_, ymax


def scale_y(_cords, resize_y):
    xmin, ymin, xmax, ymax = _cords
    ymid = (ymin + ymax) / 2
    ymin_ = math.ceil(ymid - resize_y * (ymid - ymin))
    ymax_ = math.floor(ymid + resize_y * (ymax - ymid))

    return xmin, ymin_, xmax, ymax_


def draw_box(cords, pth_img, pth_img_rect, color=(0, 0, 255), resize_x=1.0, thickness=1, text='', seqnum=False,
             hidebox=False, show_img=False):
    try:
        # img = cv2.imread(pth_img)
        img = cv_imread(pth_img)  # 解决中文路径文件的读
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        draw_1 = img

        boxes = []

        for cord in cords:
            _cord = int(cord[0]), int(cord[1]), int(cord[4]), int(cord[5])
            xmin, ymin, xmax, ymax = _cord
            if resize_x < 1.0:
                _cord = scale_x(_cord, resize_x)
            boxes.append(_cord)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for ibox, box in enumerate(boxes):
            x, y, x_, y_ = box
            # draw_1 = cv2.rectangle(img, (x,y), (x_,y_), (0,0,255), 2 )
            thick = thickness if not hidebox else 0

            draw_1 = cv2.rectangle(img, (x, y), (x_, y_), color, thick)
            if seqnum:
                draw_1 = cv2.putText(img, str(ibox), (int((x + x_) / 2 - 10), y + 10), font, 1.1, color=color,
                                     thickness=2)
            if not '' == text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                draw_1 = cv2.putText(img, text, (int((x + x_) / 2 - 30), y), font, 1, color=color, thickness=2)

        # print('Writing to image with rectangle {}\n'.format(pth_img_rect))
        cv_imwrite(pth_img_rect, draw_1)  # 解决中文路径文件的写
        if show_img:
            cv2.imshow('img', draw_1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, e.__str__(), ',File', fname, ',line ', exc_tb.tb_lineno)
        traceback.print_exc()  # debug.error(e)


def split(a):  # 获取各行起点和终点
    # b是a的非0元素的下标 组成的数组 (np格式),同时也是高度的值
    b = np.transpose(np.nonzero(a))
    # print(b,type(b))
    # print(a,b.tolist())

    star = []
    end = []
    star.append(int(b[0]))
    for i in range(len(b) - 1):
        cha_dic = int(b[i + 1]) - int(b[i])
        if cha_dic > 1:
            # print(cha_dic,int(b[i]),int(b[i+1]))
            end.append(int(b[i]))
            star.append(int(b[i + 1]))
    end.append(int(b[len(b) - 1]))
    # print(star) # [13, 50, 87, 124, 161]
    # print(end)  # [36, 73, 110, 147,184]
    return star, end

def scan_vertical_shadow(img, img_bi):  # 垂直投影+分割
    # 1.垂直投影
    h, w = img_bi.shape
    shadow_v = img_bi.copy()
    a = [0 for z in range(0, w)]
    # print(a) #a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数

    # print('h = ', h)
    # print('w = ', w)
    # 记录每一列的波峰
    for j in range(0, w):  # 遍历一列
        for i in range(0, h):  # 遍历一行
            if shadow_v[i, j] == 0:  # 如果该点为黑点(默认白底黑字)
                a[j] += 1  # 该列的计数器加一计数
                shadow_v[i, j] = 255  # 记录完后将其变为白色
                # print (j)
    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            shadow_v[i, j] = 0  # 涂黑

    return a


def bi_scan_box(mini_h, xmin, xmax, ymin, ymax, shadow_v):
    x_new = [xmin, xmax]
    w = xmax - xmin
    h = ymax - ymin
    x_mid = int(w/2)
    x_quarter = int(w/4)
    x_eighth = int(w/8)
    x_sixteenth = int(w/16)
    count = 0
    direction = -1
    # while x_new[1] - x_new[0] <= 3 :
    while count < 2:
        for j in range(1 * x_quarter, x_mid):
            line_sum = 0
            for k in range(ymin, ymax):
                # print(x_mid + direction * j, k, shadow_v[k, x_mid + direction * j])
                x_move = x_mid + direction * j
                if shadow_v[k, x_move] == 0:
                    break
                else:
                    line_sum += shadow_v[k, x_move]
            if line_sum == 255 * h:
                x_new.pop(count)
                x_new.insert(count, x_move)
                break
        direction = 1
        count += 1
        # x_mid = int(x_mid / 2)
    print(x_new)
    x1_new, x2_new = x_new[0], x_new[1]
    if x2_new - x1_new < mini_h:
        x1_new, x2_new = xmin, xmax
    return x1_new, x2_new


def bi_scan_box_whole_pic(mini_h, xmin, xmax, ymin, ymax, shadow_v):
    x_new = [xmin, xmax]
    # print('原始', x_new)
    w = xmax - xmin
    h = ymax - ymin
    x_mid_pic = int((xmin + xmax) / 2)
    x_quarter = int(w / 4)
    x_eighth = int(w / 8)
    x_sixteenth = int(w / 16)
    count = 0
    direction = -1
    # while x_new[1] - x_new[0] <= 3 :
    while count < 2:
        # for j in range(1 * x_eighth, 3 * x_quarter):
        for j in range(1 * x_eighth, 3 * x_quarter):
            line_sum = 0
            for k in range(ymin, ymax):
                # print(x_mid + direction * j, k, shadow_v[k, x_mid + direction * j])
                x_move = x_mid_pic + direction * j
                line_sum += shadow_v[k, x_move]
                # if shadow_v[k, x_move] == 0:
                #     break
                # else:
                #     line_sum += shadow_v[k, x_move]
            # if line_sum >= 255 * (h - 1):
            if line_sum >= 255 * h:
                # print(line_sum)
                x_new.pop(count)
                x_new.insert(count, x_move)
                break
            elif line_sum == 0:  # 全黑问题处理
                char_w = h
                # x_new.pop(count)
                if count == 0:
                    x_new = [xmax - char_w, xmax]
                else:
                    x_new = [xmin, xmin + char_w]
                shadow_v_np = np.array(shadow_v)
                if np.count_nonzero(shadow_v_np):
                    return [0, 0]
                break
        direction = 1
        count += 1
        # x_mid = int(x_mid / 2)
    x1_new, x2_new = x_new[0], x_new[1]
    H = min(mini_h, 70)
    if (x2_new - x1_new < H) or (h / (x2_new - x1_new) <= 1.2):  # w 54 h 59  59/54 = 1.09, w 71 h 61 h/w = 1.16
        x1_new, x2_new = xmin, xmax
    # print(x_new, x2_new - x1_new, mini_h)
    # print('返回值', x1_new, x2_new)
    return x1_new, x2_new


def box_scan_vertical(shadow_v, mini_h, xmin, xmax, ymin, ymax):  # 扫描得到最小左右框
    # h, w = img_bi.shape
    # h = ymax - ymin
    # shadow_v = img_bi.copy()
    # a = [0 for z in range(0, w)]
    # x_mid = int(w / 2)
    # for i in range(0, h):
    #     if shadow_v[x_quarter1, i] == 0:
    #         direction = -1
    # x1_new, x2_new = bi_scan_box(mini_h, xmin, xmax, ymin, ymax, shadow_v)
    x1_new, x2_new = bi_scan_box_whole_pic(mini_h, xmin, xmax, ymin, ymax, shadow_v)
    # print('box_scan_vertical', x1_new, x2_new)
    # while x2_new - x1_new <= 2:
    #     w = int(w / 2)
    #     x1_new, x2_new = bi_scan_box(h, w, shadow_v)
    return x1_new, x2_new


# def scan_margin(boxes_list=[], ):
def scan_minibox(boxes_list=[], img=''):
    # 读取要被切割的图片
    img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bi = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)
    shadow_v = img_bi.copy()
    h_list = []
    for boxes in boxes_list:
        h_list.append(boxes[5] - boxes[1])
        # _cord = xmin, ymin, xmax, ymax = int(boxes[0]), int(boxes[1]), int(boxes[4]), int(boxes[5])

    mini_h = min(h_list)
    # print(h_list)

    new_boxes_list = []
    for boxes in boxes_list:
        _cord = xmin, ymin, xmax, ymax = int(boxes[0]), int(boxes[1]), int(boxes[4]), int(boxes[5])
        # # 要被切割的开始的像素的高度值
        # beH = 60
        # # 要被切割的结束的像素的高度值
        # hEnd = 232
        # # 要被切割的开始的像素的宽度值
        # beW = 43
        # # 要被切割的结束的像素的宽度值
        # wLen = 265
        # # 对图片进行切割
        # dstImg = img_gray[ymin:ymax, xmin:xmax]
        # dstImg_bi = img_bi[ymin:ymax, xmin:xmax]
        # x1_, x2_ = box_scan_vertical(dstImg, dstImg_bi, mini_h)  # 输入图片 和 二值图, 即可进行字符分割
        x1_, x2_ = box_scan_vertical(shadow_v, mini_h, xmin, xmax, ymin, ymax)  # 输入图片 和 二值图, 即可进行字符分割
        # new_x1, new_x2 = xmin + x1_, xmin + x2_
        new_x1, new_x2 = x1_, x2_
        # print(new_x1, new_x2)
        if not new_x1 == new_x2 == 0:
            new_box = [new_x1, ymin, new_x2, ymin, new_x2, ymax, new_x1, ymax]
            new_boxes_list.append(new_box)
        # 展示原图
        # cv2.imshow("img", img)
        # 展示切割好的图片
        # dstImg_mini = img_gray[ymin:ymax, new_x1:new_x2]
        # cv2.imshow("dstImg", dstImg_mini)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # if not (new_x1 == None and new_x2 == None):
        # if not (new_x1 == new_x2):
        #     展示切割好的图片
            # dstImg_mini = img_gray[ymin:ymax, new_x1:new_x2]
            # cv2.imshow("dstImg", dstImg_mini)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # new_box = [new_x1, ymin, new_x2, ymin, new_x2, ymax, new_x1, ymax]
            # new_boxes_list.append(new_box)

        # else:
        #     new_boxes_list.append(boxes)
    # print(new_boxes_list)

    return new_boxes_list


# 解决两字一框问题
def flat_box_processor(boxes_list=[]):
    new_boxes_list = []
    for boxes in boxes_list:
        _cord = xmin, ymin, xmax, ymax = int(boxes[0]), int(boxes[1]), int(boxes[4]), int(boxes[5])
        w = xmax - xmin
        h = ymax - ymin
        w_h = w / h
        thres = 0.2
        '''
        相关参考 109/65 = 1.67
        '''
        if w_h > 1.6:
            # adjust_x = (w - h * thres) / 2  # 右边框一般会多出来一点, 稍微减去一点, 默认是框了两个字, 0.3 应该是阈值, 174图此处接近0.5
            adjust_x = w / 2
            left_box_x1, left_box_x2 = xmin, xmin + adjust_x  # 左框在后
            # right_box_x1, right_box_x2 = xmax - h * thres - adjust_x, xmax - h * thres
            right_box_x1, right_box_x2 = xmax - adjust_x, xmax
            left_box = [left_box_x1, ymin, left_box_x2, ymin, left_box_x2, ymax, left_box_x1, ymax]
            right_box = [right_box_x1, ymin, right_box_x2, ymin, right_box_x2, ymax, right_box_x1, ymax]
            new_boxes_list.append(right_box)
            new_boxes_list.append(left_box)
        else:
            new_boxes_list.append(boxes)

    return new_boxes_list


def con_line_boxes(boxes_list=[]):
    flag = True
    new_boxes_list = []
    # boxes_list.sort(key=lambda pt: pt[0], reverse=True)
    for i in range(1, len(boxes_list)):
        # for boxes in boxes_list:
        _cord = xmin, ymin, xmax, ymax = int(boxes_list[i - 1][0]), int(boxes_list[i - 1][1]), int(
            boxes_list[i - 1][4]), int(boxes_list[i - 1][5])
        # _cord = xmin, ymin, xmax, ymax = int(boxes_list[i][0]), int(boxes_list[i][1]), int(boxes_list[i][4]), int(boxes_list[i][5])
        # for j in range(i, len(boxes_list)):
        #     if not j == i:
        _cord_next = xmin2, ymin2, xmax2, ymax2 = int(boxes_list[i][0]), int(boxes_list[i][1]), int(
            boxes_list[i][4]), int(boxes_list[i][5])
        # _cord_next = xmin2, ymin2, xmax2, ymax2 = int(boxes_list[j][0]), int(boxes_list[j][1]), int(boxes_list[j][4]), int(boxes_list[j][5])
        if (abs(xmin2 - xmin) < 15) and (abs(xmax2 - xmax) < 15) and (
                abs(ymax2 - ymin) < 200 or abs(ymax - ymin2) < 200):
            x1, x2, y1, y2 = min(xmin, xmin2), min(xmax, xmax2), min(ymin, ymin2), max(ymax, ymax2)  # 这里右边用min x
            new_box = [x1, y1, x2, y1, x2, y2, x1, y2]
            new_boxes_list.append(new_box)
            flag = False
        else:
            # new_boxes_list.append(boxes_list[i - 1])
            if flag:
                new_boxes_list.append(boxes_list[i - 1])
            else:
                flag = True
    new_boxes_list.append(boxes_list[-1])

    return new_boxes_list


def con_line_boxes_two_point(boxes_list=[]):
    flag = True
    new_boxes_list = []
    # boxes_list.sort(key=lambda pt: pt[0], reverse=True)
    for i in range(1, len(boxes_list)):
        # for boxes in boxes_list:
        _cord = xmin, ymin, xmax, ymax = int(boxes_list[i - 1][0]), int(boxes_list[i - 1][1]), int(
            boxes_list[i - 1][2]), int(boxes_list[i - 1][3])
        # _cord = xmin, ymin, xmax, ymax = int(boxes_list[i][0]), int(boxes_list[i][1]), int(boxes_list[i][4]), int(boxes_list[i][5])
        # for j in range(i, len(boxes_list)):
        #     if not j == i:
        _cord_next = xmin2, ymin2, xmax2, ymax2 = int(boxes_list[i][0]), int(boxes_list[i][1]), int(
            boxes_list[i][2]), int(boxes_list[i][3])
        # _cord_next = xmin2, ymin2, xmax2, ymax2 = int(boxes_list[j][0]), int(boxes_list[j][1]), int(boxes_list[j][4]), int(boxes_list[j][5])
        if (abs(xmin2 - xmin) < 15) and (abs(xmax2 - xmax) < 30) \
                and (abs(ymax2 - ymin) < 250 or abs(ymax - ymin2) < 250):
            x1, x2, y1, y2 = min(xmin, xmin2), max(xmax, xmax2), min(ymin, ymin2), max(ymax, ymax2)  # 这里右边用min x
            new_box = [x1, y1, x2, y2]
            new_boxes_list.append(new_box)
            flag = False
        else:
            # new_boxes_list.append(boxes_list[i - 1])
            if flag:
                new_boxes_list.append(boxes_list[i - 1])
            else:
                flag = True
    new_boxes_list.append(boxes_list[-1])

    return new_boxes_list


def con_line_boxes_test(boxes_list=[]):
    flag = False
    new_boxes_list = boxes_list
    # boxes_list.sort(key=lambda pt: pt[0], reverse=True)
    if len(new_boxes_list) > 1:
        while not flag:
            for i in range(1, len(new_boxes_list)):
                # for boxes in boxes_list:
                _cord = xmin, ymin, xmax, ymax = int(new_boxes_list[i - 1][0]), int(new_boxes_list[i - 1][1]), int(
                    new_boxes_list[i - 1][4]), int(new_boxes_list[i - 1][5])
                # _cord = xmin, ymin, xmax, ymax = int(boxes_list[i][0]), int(boxes_list[i][1]), int(boxes_list[i][4]), int(boxes_list[i][5])
                # for j in range(i, len(boxes_list)):
                #     if not j == i:
                _cord_next = xmin2, ymin2, xmax2, ymax2 = int(new_boxes_list[i][0]), int(new_boxes_list[i][1]), int(
                    new_boxes_list[i][4]), int(new_boxes_list[i][5])
                # _cord_next = xmin2, ymin2, xmax2, ymax2 = int(boxes_list[j][0]), int(boxes_list[j][1]), int(boxes_list[j][4]), int(boxes_list[j][5])
                if (abs(xmin2 - xmin) < 15) and (abs(xmax2 - xmax) < 15) and (
                        (abs(ymax2 - ymin) < 100) or (abs(ymax - ymin2) < 100)):
                    x1, x2, y1, y2 = min(xmin, xmin2), max(xmax, xmax2), min(ymin, ymin2), max(ymax,
                                                                                               ymax2)  # 这里右边用min x
                    new_box = [x1, y1, x2, y1, x2, y2, x1, y2]
                    new_boxes_list.pop(i - 1)
                    new_boxes_list.pop(i - 1)
                    new_boxes_list.insert(i - 1, new_box)
                    flag = False
                    break
                else:
                    flag = True
                    # new_boxes_list.append(boxes_list[i - 1])
                    # if flag:
                    #     new_boxes_list.append(boxes_list[i - 1])
                    # else:
                    #     flag = True
        # new_boxes_list.append(boxes_list[-1])

    return new_boxes_list


def minimize_box(boxes_list=[], img=''):
    # 读取要被切割的图片
    img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bi = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)

    new_boxes_list = []
    flag = 0
    for boxes in boxes_list:
        flag += 1
        _cord = xmin, ymin, xmax, ymax = int(boxes[0]), int(boxes[1]), int(boxes[4]), int(boxes[5])
        # # 要被切割的开始的像素的高度值
        # beH = 60
        # # 要被切割的结束的像素的高度值
        # hEnd = 232
        # # 要被切割的开始的像素的宽度值
        # beW = 43
        # # 要被切割的结束的像素的宽度值
        # wLen = 265
        # # 对图片进行切割
        dstImg = img_gray[ymin:ymax, xmin:xmax]
        dstImg_bi = img_bi[ymin:ymax, xmin:xmax]
        x1_, x2_ = box_cut_vertical(dstImg, dstImg_bi)  # 输入图片 和 二值图, 即可进行字符分割

        if not ((x1_ == x2_) or (x1_ == None) or (x2_ == None)):
            new_x1, new_x2 = xmin + x1_, xmin + x2_
            # 展示原图
            # cv2.imshow("img", img)
            # 展示切割好的图片
            # dstImg_mini = img_gray[ymin:ymax, new_x1:new_x2]
            # cv2.imshow("dstImg", dstImg_mini)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # if not (new_x1 == None and new_x2 == None):
            # 展示切割好的图片
            # dstImg_mini = img_gray[ymin:ymax, new_x1:new_x2]
            # cv2.imshow("dstImg", dstImg_mini)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            new_box = [new_x1, ymin, new_x2, ymin, new_x2, ymax, new_x1, ymax]
            new_boxes_list.append(new_box)
        # else:
        #     new_boxes_list.append(boxes)

    return new_boxes_list


def boxes_processor(res4api_detect_line, pth_img='', dbg=False):
    # def concat_boxes(res4api_detect_line, res4api_detect_line_db, pth_img='', dbg=False):
    filename, pth_sav_dir = '', ''
    if not '' == pth_img:
        filename, file_ext = os.path.splitext(os.path.basename(pth_img))
        pth_dir = os.path.abspath(os.path.dirname(pth_img))
        pth_sav_dir = os.path.join(pth_dir, 'output')

    # 〇.分别获取craft和db的api结果
    # 画craft框（红色）  BEGIN
    res_detect_line = {
        int(itm['name']): {'box': [float(pt) for pt in itm['box']], 'text': itm['text']} for itm in res4api_detect_line
    }
    out_sav = ''
    cords_craft_orig = [v['box'] for index, v in res_detect_line.items()]
    cords_craft_orig_backup = [v['box'] for index, v in res_detect_line.items()]

    cords_craft_orig = minimize_box(cords_craft_orig, pth_img)
    cords_craft_orig = flat_box_processor(cords_craft_orig)
    cords_craft_orig = scan_minibox(cords_craft_orig, pth_img)
    # print(cords_craft_orig)
    # cords_craft_orig = con_line_boxes_test(cords_craft_orig)
    # cords_craft_orig = con_line_boxes(cords_craft_orig)
    # cords_craft_orig = con_line_boxes(cords_craft_orig)  # 目前来看少做一次会丢框

    pth_img_rect = os.path.join(pth_sav_dir, filename + 'rec.jpg') if not '' == pth_img else ''
    #
    # # 在craft画框基础上，再画db框（蓝色）  BEGIN
    # res_detect_line_db = {
    #     int(itm['name']): {'box': [float(pt) for pt in itm['box']], 'text': itm['text']} for itm in
    # res4api_detect_line_db
    # }
    # cords_db_orig = [v['box'] for index, v in res_detect_line_db.items()]

    # I. 将坐标加上宽,长,宽长比等信息
    cords_craft_orig_ = [conv_cords(cord) for cord in cords_craft_orig]
    # cords_craft_orig_backup_ = [conv_cords(cord) for cord in cords_craft_orig_backup]
    # cords_db_orig_ = [conv_cords(cord) for cord in cords_db_orig]
    # cords_orig_ = cords_craft_orig_ + cords_db_orig_
    cords_orig_ = cords_craft_orig_

    # 过滤 宽w/长h > 1.1的框， RATIO_WH_FILTER 参数配置
    # RATIO_WH_FILTER = 1.1  # 过滤 w/h > 1.05 ?  的框
    # RATIO_WH_FILTER = 1.5  # 过滤 w/h > 1.05 ?  的框 原宽长比1.1,拟改成1.5

    # W_AVG = np.mean([cord[-3] for cord in cords_orig_])
    # H_AVG = np.mean([cord[-2] for cord in cords_orig_])
    # # 改变初始过滤策略:过滤交叉>=3的框 -- BEGIN
    # # cords_orig_ = [ cord for cord in cords_orig_ if cord[-1]<=RATIO_WH_FILTER and cord[-2]>W_AVG]
    # cords_orig_ = [cord for cord in cords_orig_ if cord[-1] <= RATIO_WH_FILTER]

    # if dbg and not pth_img=='':
    if not pth_img == '':
        # 画craft框（红色）  END
        # print(cord_craft_orig for cord_craft_orig in cords_craft_orig)
        for i in range(len(cords_craft_orig)):
            print(i, cords_craft_orig[i])
        draw_box(cords_craft_orig, pth_img, pth_img_rect, seqnum=True)
        # 在优化框基础上，再画原craft框（蓝色）  END
        draw_box(cords_craft_orig_backup, pth_img_rect, pth_img_rect, color=(255, 0, 0))

    # 改变初始过滤策略:过滤交叉>=3的框 -- BEGIN
    _Xmin = min([c_d[0] for c_d in cords_orig_])
    _Ymin = min([c_d[1] for c_d in cords_orig_])
    _minDot = (_Xmin, _Ymin)  # 最左上的点

    # cords_inter2plus = [cord for cord in cords_craft_orig_ if isinter2plus(cord, cords_db_orig_, _minDot)]
    # for cord in cords_inter2plus:
    #     try:
    #         cords_orig_.remove(cord)
    #     except Exception as e:
    #         continue
    # # 改变初始过滤策略:过滤交叉>=3的框 -- END

    # II. 坐标转换(宽度缩放到0.7)
    cords, cords_craft, cords_db = [], [], []
    # cords, cords_craft = [], []
    # 宽度缩放比例， RESIZE_X 参数配置
    RESIZE_X, RESIZE_Y = 0.9, 1

    for cord in cords_orig_:  # x方向坐标压缩（避免横向不同大框之间的框交错）
        _cord = xmin, ymin, xmax, ymax = cord[0], cord[1], cord[2], cord[3]
        if RESIZE_X < 1.0:
            _cord = list(scale_x(_cord, RESIZE_X))
            _cord = list(scale_y(_cord, RESIZE_Y))
        cords.append(_cord)
    cords_craft = cords
    # cords_db = cords
    # cords = cords_orig_

    # for cord_cft in cords_craft_orig:
    #     _cord_cft = int(cord_cft[0]), int(cord_cft[1]), int(cord_cft[4]), int(cord_cft[5])
    #     if RESIZE_X < 1.0:
    #         _cord_cft = list(scale_x(_cord_cft, RESIZE_X))
    #         _cord_cft = list(scale_y(_cord_cft, RESIZE_Y))
    #     cords_craft.append(_cord_cft)
    cords_craft = cords_craft_orig

    # for cord_db in cords_db_orig:
    #     _cord_db = int(cord_db[0]), int(cord_db[1]), int(cord_db[4]), int(cord_db[5])
    #     if RESIZE_X < 1.0:
    #         _cord_db = list(scale_x(_cord_db, RESIZE_X))
    #         _cord_db = list(scale_y(_cord_db, RESIZE_Y))
    #     cords_db.append(_cord_db)
    # cords_db = cords

    # 求全局XMIN, YMIN, XMAX, YMAX
    Xmin = min([c_d[0] for c_d in cords])
    Ymin = min([c_d[1] for c_d in cords])
    Xmax = max([c_d[2] for c_d in cords])
    Ymax = max([c_d[3] for c_d in cords])

    # line [xmin, Ymin-2, xmax, Ymin-2]
    lines = [[c_d[0], Ymin - 10, c_d[2], Ymin - 10] for c_d in cords]
    # 计算线段平均长度
    line_lens = [abs(l[2] - l[0]) for l in lines]
    LLEN_AVG = np.mean(line_lens)
    RATIO_LINE_FILTER = 2 / 3
    # 过滤掉过短的线段
    lines = [l for l in lines if (LLEN_AVG * RATIO_LINE_FILTER) < (l[2] - l[0])]
    lines.sort(key=lambda pt: pt[2], reverse=True)

    # if dbg and not pth_img=='':
    if not pth_img == '':
        # 画craft框（红色）  END
        _cords_craft = [[c[0], c[1], c[2], c[1], c[2], c[3], c[0], c[3]] for c in cords_craft]
        _cords_db = [[c[0], c[1], c[2], c[1], c[2], c[3], c[0], c[3]] for c in cords_db]
        pth_img_rect_resize = os.path.join(pth_sav_dir, filename + 'rec_resize.jpg') if not '' == pth_img else ''
        draw_box(cords_craft_orig, pth_img, pth_img_rect_resize, seqnum=True)
        # 在craft画框基础上，再画db框（蓝色）  END
        # draw_box(_cords_db, pth_img_rect_resize, pth_img_rect_resize, color=(255, 0, 0))

    # III. x轴方向线段归并
    lines_union, line_u = [], lines[0]

    for i in range(1, len(lines)):
        _line = lines[i]
        interacted, _line_u, _ = line_interact(line_u, _line, minDot=(Xmin, Ymin))
        if interacted:
            line_u = _line_u
        else:
            lines_union.append(line_u)
            line_u = _line
    lines_union.append(line_u)

    # 画线段
    if dbg:
        pth_img_with_unionline = pth_img_rect.replace('rec.jpg', 'rec_uline.jpg')
        draw_line(lines_union, pth_img_rect, pth_img_with_unionline, color=(0, 255, 0), thickness=3)

    # IV. 对于归并后线段，求间隙中竖线位置，从而求得大框（大行）坐标
    # 并将归并后线段、中竖线绘制在图片上
    bigboxes = []
    bigboxes_points = [[(lines_union[i - 1][0] + lines_union[i][2]) / 2, Ymin - 5] for i in range(1, len(lines_union))]
    # 最右边一个点 + [...] + 最左边一个点
    bigboxes_points = [[lines_union[0][2] + 16, Ymin - 5]] + bigboxes_points + [[lines_union[-1][0] - 16, Ymin - 5]]
    for i in range(1, len(bigboxes_points)):
        _xmin, _ymin = bbp = bigboxes_points[i]
        _xmax, _ymin = bbp_pre = bigboxes_points[i - 1]
        bigbox = [_xmin + 4, _ymin, _xmax - 4, _ymin, _xmax - 4, Ymax + 5, _xmin + 4, Ymax + 5]
        bigboxes.append(bigbox)

    # 画大框
    if dbg:
        pth_img_with_bigbox = pth_img_rect.replace('rec.jpg', 'rec_bigbox.jpg')
        draw_box(bigboxes, pth_img_rect_resize, pth_img_with_bigbox, color=(0, 128, 0), thickness=2, seqnum=True)

    # V. 过滤属于大框内的subbox
    cords.sort(key=lambda pt: pt[0], reverse=True)
    # cords = con_line_boxes_two_point(cords)
    # cords = con_line_boxes_two_point(cords)
    # cords = con_line_boxes_two_point(cords)

    # print(cords)
    bigboxes_subboxes = {
        i: {
            'cords_big': bigbox,  # 四点坐标
            'sub_boxes': filter_subbox(cords, bigbox, minDot=(Xmin, Ymin))  # 左上右下两点坐标
        }
        for i, bigbox in enumerate(bigboxes)
    }
    # 生成随机颜色数组
    colors_box = randcolors(len(bigboxes_subboxes))
    if dbg:
        # BEGIN - 重新画大框 和 大框包含的子框 （大框和子框颜色相同，大框编号）
        # print(bigboxes_subboxes)
        # 画大框、标数字
        pth_img_subbox = pth_img_rect.replace('rec.jpg', 'rec_subbox.jpg')
        pth_img_ubox = pth_img_rect.replace('rec.jpg', 'rec_ubox.jpg')

        draw_box([bigboxes_subboxes[0]['cords_big']], pth_img, pth_img_subbox, color=colors_box[0], thickness=1,
                 seqnum=True,
                 text='0')
        for i, bigitem in bigboxes_subboxes.items():
            bigbox = bigitem['cords_big']
            _sub_boxes = bigitem['sub_boxes']
            # drawbox
            sub_boxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in _sub_boxes]
            color = colors_box[i]
            if i > 0:
                draw_box([bigbox], pth_img_subbox, pth_img_subbox, color=color, thickness=1, seqnum=True, text=str(i))
            draw_box(sub_boxes, pth_img_subbox, pth_img_subbox, color=color, seqnum=True, thickness=1)
        # END - 重新画大框 和 大框包含的子框 （大框和子框颜色相同，大框编号）

    # VI. 各大框内的subbox融合归并
    # bigitem['sub_boxes'] = con_line_boxes_two_point(bigitem['sub_boxes'])
    # bigitem['sub_boxes'] = con_line_boxes_two_point(bigitem['sub_boxes'])
    for i, bigitem in bigboxes_subboxes.items():
        sub_boxes = bigitem['sub_boxes']
        uboxes = union_subboxes(sub_boxes, cords_db, minDot=(Xmin, Ymin))
        bigboxes_subboxes[i]['uboxes'] = uboxes

    color_ubox = (0, 128, 0)
    if dbg:
        # BEGIN - 重新画大框 和 大框包含的融合后子框 （大框和子框颜色相同，大框编号）
        # 画大框、标数字
        pth_img_ubox = pth_img_rect.replace('rec.jpg', 'rec_ubox.jpg')

        # draw_box([ bigboxes_subboxes[0]['cords_big'] ], pth_img, pth_img_ubox, color=colors_box[0], thickness=1, text='0')
        draw_box([bigboxes_subboxes[0]['cords_big']], pth_img, pth_img_ubox, color=color_ubox, seqnum=True, thickness=1,
                 text='0')
        for i, bigitem in bigboxes_subboxes.items():
            bigbox = bigitem['cords_big']
            _uboxes = bigitem['uboxes']
            # drawbox
            uboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in _uboxes]
            color = color_ubox  # colors_box[i]
            if i > 0:
                draw_box([bigbox], pth_img_ubox, pth_img_ubox, color=color, thickness=1, seqnum=True, text=str(i))
            draw_box(uboxes, pth_img_ubox, pth_img_ubox, color=color, thickness=1, seqnum=True)
        # END - 重新画大框 和 大框包含的融合后子框 （大框和子框颜色相同，大框编号）

    for i, bigitem in bigboxes_subboxes.items():
        bigbox = bigitem['cords_big']
        _uboxes = bigitem['uboxes']
        # 7.1 竖向分组, 竖向组内，横向排序
        ygrp_uboxes, ulines_y = ygroup_uboxes(_uboxes, minDot=(Xmin, Ymin))
        bigboxes_subboxes[i]['ygrp_uboxes'] = ygrp_uboxes
        bigboxes_subboxes[i]['ulines_y'] = ulines_y

    # 绘制每个bigbox框下，y轴方向 ulines_y
    # pth_img_ubox
    ulinesy_all = [uliney for i, bigitem in bigboxes_subboxes.items() for uliney in bigitem['ulines_y']]

    if dbg:
        pth_img_uliney = pth_img_rect.replace('rec.jpg', 'rec_ulinesy.jpg')
        draw_line(ulinesy_all, pth_img_ubox, pth_img_uliney, color=(0, 255, 0), thickness=3)

    # VII. uboxes 整体编号
    idx_ubox_g = 0
    uboxes_g, bigboxes_uboxes = [], {}
    for i, bigitem in bigboxes_subboxes.items():
        bigbox = bigitem['cords_big']
        _uboxes = bigitem['uboxes']
        ygrp_uboxes = bigitem['ygrp_uboxes']
        bigboxes_subboxes[i]['uboxes_g'] = {}
        for j, ygrp_ubox in ygrp_uboxes.items():
            # jiapi = 'normal'
            # if len(ygrp_ubox) > 1: jiapi='jiapi'
            for _ubox in ygrp_ubox:
                # x方向缩放10/7
                # _ubox = scale_x(_ubox, resize_x=(10 / 6))
                _ubox = scale_x(_ubox, resize_x=(10 / 9))
                # y方向缩放10/9
                # _ubox = scale_y(_ubox, resize_y=(10 / 9))
                # _ubox = scale_y(_ubox, resize_y=(10 / 10))
                bigboxes_subboxes[i]['uboxes_g'][idx_ubox_g] = _ubox
                idx_ubox_g += 1

                # uboxes_g.append( (_ubox, jiapi) )
                uboxes_g.append(_ubox)
        bigboxes_uboxes[i] = {
            'cords_big': bigbox,
            'uboxes_g': bigboxes_subboxes[i]['uboxes_g']
        }

    res4api_detect_line_union = {
        'uboxes_g': uboxes_g,  # 所有ubox全部放进来
        'bigboxes_uboxes': bigboxes_uboxes
    }

    # 绘制uboxes_g到图片上（测试史记、缺漏图）
    # 查看新db的可视化，数字正好在矩形中间，字体大小合适
    uboxes_lurd = [[
        ub[0], ub[1], ub[2], ub[1], ub[2], ub[3], ub[0], ub[3]
        # ] for (ub,jiapi) in uboxes_g]
    ] for ub in uboxes_g]

    # 把bigbox用暗红色画出来，画在uboxes_g上（对于没有双行夹批的，大框替代所有小框并在x,y方向缩放0.96）
    bigboxes = [bigitem['cords_big'] for i, bigitem in bigboxes_subboxes.items()]

    if not '' == pth_img:
        pth_img_uboxes_g = pth_img_rect.replace('rec.jpg', 'rec_uboxes_g.jpg')
        draw_box(uboxes_lurd, pth_img, pth_img_uboxes_g, color=(0, 0, 255), thickness=1, seqnum=True)
        # draw_box(uboxes_lurd, pth_img, pth_img_uboxes_g, color=color_ubox, thickness=3, seqnum=True)
        # draw_box(bigboxes, pth_img_uboxes_g, pth_img_uboxes_g, color=(128, 0, 0), thickness=1, seqnum=True)

    # VIII. 求框中位数，计算字体M, S（双行夹批）

    return res4api_detect_line_union

    # db和craft相交,保留db(例外: craft全包db且远长于db)
    # 重叠矩形(可能是ubox) 判断逻辑


def test_one(pth_img, dbg=False):
    img = readPILImg(pth_img)

    filename, file_ext = os.path.splitext(os.path.basename(pth_img))
    pth_dir = os.path.abspath(os.path.dirname(pth_img))
    pth_sav_dir = os.path.join(pth_dir, 'output')

    if not os.path.exists(pth_sav_dir):
        os.makedirs(pth_sav_dir)

    pth_sav = os.path.join(pth_sav_dir, filename + '_res_recog.txt')
    save_folder = os.path.join(pth_sav_dir, filename)

    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # if os.path.exists(pth_sav):
    #     print('{} already exists'.format(pth_sav))
    #     return

    imgbase64 = base64of_img(pth_img)

    # param_detect = {
    #     'picstr': imgbase64,
    #     'mod': 'base64',
    #     'do_ocr': 0
    # }``
    # r = requests.post(url_line_detect, data=param_detect)
    _auth = ('jihe.com', 'DIY#2020')
    param_recog = {
        'picstr': imgbase64,
        'mod': 'base64'
    }

    img = readPILImg(pth_img)
    # 0 craft_字
    res4api_detect_line = request_api(url_page_recog_0, param_recog, _auth)
    # 1 DB 测试
    # res4api_detect_line = request_api(url_page_recog_1, param_recog, _auth)
    # 2 craft行测试
    # res4api_detect_line = request_api(url_page_recog, param_recog, _auth)
    # print(res4api_detect_line)

    # res_detect_line = {
    #     int(itm['name']): {'box': [float(pt) for pt in itm['box']], 'text': itm['text']} for itm in res4api_detect_line
    # }
    res_box = [
        [int(pt) for pt in itm['box']] for itm in res4api_detect_line
    ]
    print(res_box)
    '''
    {0: {'box': [1047.0, 32.0, 1123.0, 32.0, 1123.0, 1696.0, 1047.0, 1696.0], 'text': '濟之闢釋氏長孫之表五經唐名流如太白柳州宋'}}
    '''

    res_detect_line_list = [
        {'box': [float(pt) for pt in itm['box']], 'text': itm['text'], 'size': itm['size']} for itm in
        res4api_detect_line
    ]
    '''
    <class 'list'>: [{'box': [1047.0, 32.0, 1123.0, 32.0, 1123.0, 1696.0, 1047.0, 1696.0], 'text': '濟之闢釋氏長孫之表五經唐名流如太白柳州宋'}]
    '''

    out_sav = ''
    # cords = [v['box'] for index, v in res_detect_line.items()]
    cords_np = []
    # print(cords)

    # sorted_res_detect_line = bubble_sort(res_detect_line_list)
    # concat_res = concat_boxes(res4api_detect_line, res4api_detect_line_db, pth_img=pth_img, dbg=dbg)

    # sorted_res_detect_line = res_detect_line_list

    #
    # print(res_detect_line_list)
    # cords = [cord['box'] for cord in sorted_res_detect_line]

    # 以下接入融合算法
    concat_res = boxes_processor(res4api_detect_line, pth_img=pth_img, dbg=dbg)
    uboxes_g = concat_res['uboxes_g']
    # print(res4api_detect_line)

    bigboxes_uboxes = concat_res['bigboxes_uboxes']
    res_detect_line = {i: [ub[0], ub[1], ub[2], ub[1], ub[2], ub[3], ub[0], ub[3]] \
                       for i, ub in enumerate(uboxes_g)}
    # 返回API
    res4api_detect_line = [
        {
            'box': [str(pt) for pt in box],
            'name': str(i),
            'text': ''
        } for i, box in res_detect_line.items()
    ]
    # print(res4api_detect_line)

    widths_line = []
    for index, cord in res_detect_line.items():
        try:
            x1, y1, x2, y2, x3, y3, x4, y4 = cord
            min_x, max_x = round((x1 + x4) / 2), round((x2 + x3) / 2)
            widths_line.append(abs(max_x - min_x))
        except Exception as e:
            print(e)
            continue
        # for cord_ in cords:
        '''
        预备后续非长方形框使用
        '''
        # x1, y1, x2, y2, x3, y3, x4, y4 = cord
        # min_x, max_x = round((x1 + x4) / 2), round((x2 + x3) / 2)
        # width = cord_[2] - cord_[0]
        # cord_.append(width)
    w_sorted = sorted(widths_line)
    width_rngs = get_w_rngs(widths_line, R=0.065)  #er适合0.07
    print(w_sorted)
    print(width_rngs)

    sizes = ['S', 'M', 'L', 'XL', 'XXL', 'XXXL']
    boxgrp_size = {s: [] for s in sizes}
    for i, r_line in enumerate(res4api_detect_line):
        box_line = r_line['box']
        x1, y1, x2, y2, x3, y3, x4, y4 = [int(cord) for cord in box_line]
        min_x, max_x = round((x1 + x4) / 2), round((x2 + x3) / 2)
        min_y, max_y = round((y1 + y2) / 2), round((y3 + y4) / 2)
        width_line = abs(max_x - min_x)  # 再次得到宽度

        line_size = get_line_size(width_rngs, width_line)  # 聚出的类和该列宽度
        r_line['size'] = line_size

        boxgrp_size[line_size].append(box_line)
        print('{} - witdth:\t{}\tsize:{}'.format(i, width_line, r_line['size']))

    # line size remapping
    re_mapping_lsize(res4api_detect_line)
    sizes = ['S', 'M']
    boxgrp_size = {s: [] for s in sizes}
    for i, r_line in enumerate(res4api_detect_line):
        box_line = r_line['box']
        line_size = r_line['size']

        boxgrp_size[line_size].append(box_line)

    uboxes_lurd = [[
        ub[0], ub[1], ub[2], ub[1], ub[2], ub[3], ub[0], ub[3]
    ] for ub in uboxes_g]
    bigboxes = [bigitem['cords_big'] for i, bigitem in bigboxes_uboxes.items()]
    pth_img_rect = os.path.join(pth_sav_dir, filename + 'rec.jpg') if not '' == pth_img else ''
    pth_img_with_size = pth_img_rect.replace('rec.jpg', 'rec_uboxes_size.jpg')
    draw_box(uboxes_lurd, pth_img, pth_img_with_size, color=(0, 128, 0), thickness=2, seqnum=True)
    # draw_box(bigboxes, pth_img_with_size, pth_img_with_size, color=(128, 0, 0), thickness=2, seqnum=True)
    # draw_box(bigboxes, pth_img_with_size, pth_img_with_size, color=(0, 0, 255), thickness=2, seqnum=True)
    # 输出带size的图
    for size, boxgrp in boxgrp_size.items():
        if len(boxgrp) == 0: continue
        draw_box(boxgrp, pth_img_with_size, pth_img_with_size, color=(0, 128, 0), thickness=1, text=str(size),
                 hidebox=True)

    #
    # draw_box(uboxes_lurd, pth_img, pth_img_with_size, color=(0, 128, 0), thickness=2, seqnum=True)
    # draw_box(bigboxes, pth_img_with_size, pth_img_with_size, color=(128, 0, 0), thickness=1, seqnum=True)
    # pth_img_rect = os.path.join(pth_sav_dir, filename + 'rec_craft.jpg')
    # draw_box(cords, pth_img, pth_img_rect, seqnum=True, resize_x=0.8)
    # draw_box(cords, pth_img, pth_img_rect, seqnum=True, resize_x=0.8, show_img=True)
    '''

    for cord in cords:
        cord = np.array(cord)
        cord = cord.reshape(-1, 2)
        cords_np.append(cord)
    # cords = adjustColumeBoxes(cords)
    # cords = adjustBoxesoutput(cords_np)
    boxes = []
    # print(cords_np)
    boxes_np = cluster_sort(cords_np)
    # boxes_np = cords_np

    for box in boxes_np:
        box = box.reshape(-1, 8)
        for box_ in box.tolist():
            # print(box_)
            # for box__ in box_:
            boxes.append(box_)
    # print(boxes)
    '''

    '''
    cord_str = ''
    for cord_box in cords:
        cord_str += str(cord_box) + '\n'
    with open(os.path.join(pth_sav_dir, filename) + '_cords.txt', 'w') as f:
        f.write(str(cord_str))
    pth_img_rect = os.path.join(pth_sav_dir, filename + '_rec.jpg')
    # draw_box(boxes, pth_img, pth_img_rect, seqnum=True, resize_x=0.8)
    draw_box(cords, pth_img, pth_img_rect, seqnum=True, resize_x=0.8)
    res_txt = ''
    res_txt_size = ''
    # print(res_detect_line.get('text'))
    for line in sorted_res_detect_line:
    #     print(line['text'])
        res_txt += line['text'] + '\n'
        res_txt_size += '<' + line['size'] + '>' + line['text'] + '</' + line['size'] + '>' + '\n'
    # res_txt = [box['text'] + '\n' for box in sorted_res_detect_line]
    # print(res_txt)
    with open(os.path.join(pth_sav_dir, filename) + '.txt', 'w') as f:
        f.write(str(res_txt))
    with open(os.path.join(pth_sav_dir, filename) + '_size.txt', 'w') as f:
        f.write(str(res_txt_size))
    '''


def bubble_sort(items):
    for i in range(len(items) - 1):
        flag = False
        for j in range(len(items) - 1 - i):
            if check_order(items[j], items[j + 1]):
                # if not check_order(items.get[j], items.get[j + 1]):
                #     items.get[j], items.get[j + 1] = items.get[j + 1], items.get[j]
                items[j], items[j + 1] = items[j + 1], items[j]
                flag = True
        if not flag:
            break
    return items


def rng_interact(rng1, rng2):
    interacted = False
    xmin1, xmax1 = rng1[0], rng1[1]
    xmin2, xmax2 = rng2[0], rng2[1]
    if xmin1 <= xmax2 and xmin2 <= xmax1: interacted = True
    rng_u = [min(xmin1, xmin2), max(xmax1, xmax2)]
    return interacted, rng_u


def check_order(dict_1, dict_2):
    '''

    :param list_1:
    :param list_2:
    :return:
    True means list1 is before list2, False means list2 is before list 1
    '''
    if dict_1.get('box')[0] >= dict_2.get('box')[2]:
        return False
    elif (dict_1.get('box')[2] >= dict_2.get('box')[0]) and (dict_1.get('box')[1] < dict_2.get('box')[3]):
        return False
    else:
        return True


def convert_bbox_to_lrud(bbox):
    l = min(bbox[:, 0])
    r = max(bbox[:, 0])
    u = min(bbox[:, 1])
    d = max(bbox[:, 1])
    return l, r, u, d


def cluster_boxes(boxes, type='DBSCAN'):
    switch = {
        'DBSCAN': DBSCAN(min_samples=1, eps=7),
        'MeanShift': MeanShift(bandwidth=0.3),
        'OPTICS': OPTICS(min_samples=1, eps=20),
        'Birch': Birch(n_clusters=None)
    }
    cluster = switch[type]
    boxes_data = [[b['l'], b['r']] for b in boxes]
    boxes_data = np.array(boxes_data)
    labels = cluster.fit_predict(boxes_data)
    '''
    plt.scatter(boxes_data[:, 0], boxes_data[:, 1], s=1, c=labels)
    plt.show()
    '''
    classified_box_ids = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        classified_box_ids[label].append(idx)
    return classified_box_ids


def list_sort(box_list):
    r = [b['r'] for b in box_list]
    length = [b['r'] - b['l'] for b in box_list]
    r = np.mean(r)
    length = np.mean(length)
    return r + length


def box_sort(box):
    u = box['u']
    d = box['d']
    return (u + d) / 2


def cluster_sort(boxes):
    """
    :param boxes:
    :return: cluster then sorted boxes
        l = array[0, 0]
        r = array[1, 0]
        u = array[0, 1]
        d = array[2, 1]
    """
    boxes_lrud = []
    for id, box in enumerate(boxes):
        l, r, u, d = convert_bbox_to_lrud(box)
        boxes_lrud.append({'id': id, 'l': l, 'r': r, 'u': u, 'd': d})
    # boxes_lrud = [{'l': b[0, 0], 'r': b[1, 0], 'u': b[0, 1], 'd': b[2, 1], 'id': id} for id, b in enumerate(boxes)]
    '''
    classified_box_ids = projection_split(shape, boxes_lrud)
    classified_boxes = []
    for k in classified_box_ids.keys():
        box_ids = classified_box_ids[k]
        classified_boxes.append([boxes_lrud[box_id] for box_id in box_ids])
    '''
    classified_box_ids = cluster_boxes(boxes_lrud)
    classified_boxes = []
    for k in classified_box_ids.keys():
        box_ids = classified_box_ids[k]
        classified_boxes.append([boxes_lrud[box_id] for box_id in box_ids])
    classified_boxes = sorted(classified_boxes, key=list_sort, reverse=True)
    new_classifier_boxes = []
    for box_list in classified_boxes:
        new_classifier_boxes.append(sorted(box_list, key=box_sort, reverse=False))
    new_classifier_boxes = list(itertools.chain.from_iterable(new_classifier_boxes))
    new_classifier_boxes = [boxes[b['id']] for b in new_classifier_boxes]
    return new_classifier_boxes


# faster-rcnn char detect
def test_char_detect_1(pth_img):
    api_base_url = 'http://api.chinesenlp.com:7001'
    AUTH = ('jihe.com', 'DIY#2020')

    # url_char_detect = '{}/ocr/v1/char_detect_1'.format(api_base_url)
    url_char_detect = '{}/ocr/v1/char_detect'.format(api_base_url)
    img = readPILImg(pth_img)

    filename, file_ext = os.path.splitext(os.path.basename(pth_img))
    pth_dir = os.path.abspath(os.path.dirname(pth_img))
    pth_sav_dir = os.path.join(pth_dir, 'output')
    if not os.path.exists(pth_sav_dir):
        os.makedirs(pth_sav_dir)

    imgbase64 = base64of_img(pth_img)
    param_detect = {
        'picstr': imgbase64,
        'mod': 'base64',
        'do_ocr': 1
    }
    r = requests.post(url_char_detect, data=param_detect, auth=AUTH)
    str_res = r.text
    o_res = json.loads(str_res)
    res4api_detect_char = o_res['data']
    # 重组
    res_detect_char = {
        int(itm['name']): [float(pt) for pt in itm['box']] for itm in res4api_detect_char
    }
    print(res_detect_char)
    cords = [v for index, v in res_detect_char.items()]
    print(cords)
    print("=" * 20)
    # boxes = adjustColumeBoxes(cords)
    # print(boxes)
    # 绘矩形框
    pth_img_rect = os.path.join(pth_sav_dir, filename + 'rec.jpg')
    draw_box(cords, pth_img, pth_img_rect, show_img=True)


if __name__ == '__main__':
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000011.jpg')
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000002.jpg')
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000174.jpg')  # 断行
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000386.jpg')  # 黑方块问题,已解决
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000032.jpg')  # 融合行过于相邻
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000783.jpg', dbg=True)  # SM问题,识别时把其一融入,只有一个M
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000231.jpg')  # SM问题,识别时把其一融入,只有一个M
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000806.jpg')  # SM问题,识别时把其一融入,只有一个M
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000179.jpg')  # SM问题,识别时把其一融入,只有一个M
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000192.jpg')  # 漏字,未解决
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000434.jpg')  # SM问题
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000620.jpg')  # 小器字,识别成S问题
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/001110.jpg')  # 小器字,识别成S问题
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000385.jpg')  # 小器字,识别成S问题
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000291.jpg')  # 偏窄字,识别成S问题
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000387.jpg')  # 固字,没办法了
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000027.jpg')  # 小字未融合,未解决
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000620.jpg')  # 半边框
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000026.jpg')  # 左右结构中间断开字
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000152.jpg')  # 右框太多, S,M出错
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000230.jpg')  # 全是别成M,搁置
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000027.jpg')
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000018.jpg')
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/ER007_pure/20_19584_jpg/000024.jpg')
    # pth = '/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000018.jpg'
    # test_char_detect_1('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000002.jpg')
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000174_line8.png')  # 断行多
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000174_line11.png')  # 断行多
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000755.jpg')  # list_bug
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000378.jpg')  # list_bug
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000752.jpg', dbg=True)  # list_bug
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/29_19457_jpg/000043.jpg', dbg=True)
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/29_19457_jpg/000064.jpg', dbg=True)  #  顺序问题
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/29_19457_jpg/000127.jpg', dbg=True)  #  顺序问题
    # test_one('/Users/Beyoung/Desktop/Projects/corpus/DingXiu/0A0CBAE0046F4AB7BCBFE12789547A78/000009.png')  #  顺序问题
    # test_one('/Users/Beyoung/Desktop/Projects/qianpai/book_pages/imgs_vertical/book_page_82.jpg', dbg=True)  #  顺序问题
    test_one('/Users/Beyoung/Desktop/Projects/corpus/diaolong/雕龙pic/道藏/D01C6C8C_D5B1_46AE_A16A_DEADF2A1846A/280418a.tif', dbg=True)  #  顺序问题

