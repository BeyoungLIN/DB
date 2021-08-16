# -*- coding: utf-8 -*-
# @Time   : 2021/8/15 11:32
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : craft_char.py.py


import base64
import json
import math
import os
import traceback
# from skimage import io
from typing import List

import cv2
import numpy as np
import requests
import six
from PIL import Image

from zhtools.langconv import *

url_line_detect = 'http://api.chinesenlp.com:7001/ocr/v1/line_detect'
url_line_recog = 'http://api.chinesenlp.com:7001/ocr/v1/line_recog'

url_page_recog_0 = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog_0'
url_page_recog = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog'
url_page_recog_1 = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog_1'


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
                draw_1 = cv2.putText(img, str(ibox), (int((x + x_) / 2 - 10), y + 20), font, 0.6, color=color)
            if not '' == text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                draw_1 = cv2.putText(img, text, (int((x + x_) / 2 - 10), y + 16), font, 0.4, color=color)

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


def test_one(pth_img):
    img = readPILImg(pth_img)

    filename, file_ext = os.path.splitext(os.path.basename(pth_img))
    pth_dir = os.path.abspath(os.path.dirname(pth_img))
    pth_sav_dir = os.path.join(pth_dir, 'output')

    if not os.path.exists(pth_sav_dir):
        os.makedirs(pth_sav_dir)

    pth_sav = os.path.join(pth_sav_dir, filename + '_res_recog.txt')
    save_folder = os.path.join(pth_sav_dir, filename)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
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

    # res4api_detect_line = request_api(url_page_recog_0, param_recog, _auth)
    # res4api_detect_line = request_api(url_page_recog_1, param_recog, _auth)
    res4api_detect_line = request_api(url_page_recog, param_recog, _auth)

    res_detect_line = {
        int(itm['name']): {'box': [float(pt) for pt in itm['box']], 'text': itm['text']} for itm in res4api_detect_line
    }
    out_sav = ''
    cords = [v['box'] for index, v in res_detect_line.items()]
    # cords_np = []
    # print(cords)
    # for cord in cords:
    #     cord = np.array(cord)
    #     cord = cord.reshape(-1, 2)
    #     cords_np.append(cord)
    # # cords = adjustColumeBoxes(cords)
    # boxes = []
    # # print(cords_np)
    # boxes_np = adjustColumeBoxes(cords_np)
    # for box in boxes_np:
    #     box = box.reshape(-1, 8)
    #     for box_ in box.tolist():
    #         print(box_)
    #         # for box__ in box_:
    #         boxes.append(box_)
    # print(boxes)

    for i in range(len(cords)):



    pth_img_rect = os.path.join(pth_sav_dir, filename + 'rec.jpg')
    draw_box(cords, pth_img, pth_img_rect, seqnum=True, resize_x=0.8, show_img=True)

    # print(res_detect_line.get('text'))
    for box in res4api_detect_line:
        print(box.get('text'))


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

def adjustBoxesoutput(boxes):
    new_boxes = []
    vis = [False for _ in range(len(boxes))]
    for i in range(len(boxes)):
        if vis[i]:
            continue
        cur_box = boxes[i]
        flag = True
        while flag:
            flag = False
            for j in range(i + 1, len(boxes)):
                if vis[j]:
                    continue
                if cal_IoU(cur_box, boxes[j]) > row_threshold and check_vertical_dis(cur_box, boxes[j]):
                    vis[j] = True
                    cur_box = mergeArray(cur_box, boxes[j])
                    flag = True
        vis[i] = True
        new_boxes.append(cur_box)

def adjustColumeBoxes(boxes: List, row_threshold=0.67, col_threshold=1.35, union_threshold=0.8):
    def cal_IoU(array1, array2):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array2_l = array2[0, 0]
        array2_r = array2[1, 0]
        in_l = max(array1_r, array2_r) - min(array1_l, array2_l)
        # in_l = max(0, in_l)
        un_l = min(array1_r, array2_r) - max(array1_l, array2_l)
        un_l = max(0, un_l)
        return un_l / in_l

    def check_vertical_dis(array1, array2):
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array2_u = array2[0, 1]
        array2_d = array2[2, 1]
        dis = max(array1_u, array2_u) - min(array1_d, array2_d)
        array2_ver = array2_d - array2_u
        if dis < array2_ver * col_threshold:
            return True
        else:
            return False

    def mergeArray(array1, array2):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array2_l = array2[0, 0]
        array2_r = array2[1, 0]
        array2_u = array2[0, 1]
        array2_d = array2[2, 1]
        array1_area = (array1_r - array1_l) * (array1_d - array1_u)
        array2_area = (array2_r - array2_l) * (array2_d - array2_u)
        new_array_l = (array1_l * array1_area + array2_l * array2_area) / (array1_area + array2_area)
        new_array_r = (array1_r * array1_area + array2_r * array2_area) / (array1_area + array2_area)
        new_array_u = min(array1_u, array2_u)
        new_array_d = max(array1_d, array2_d)
        new_array = [[new_array_l, new_array_u], [new_array_r, new_array_u],
                     [new_array_r, new_array_d], [new_array_l, new_array_d]]
        new_array = np.array(new_array, dtype=np.float)
        return new_array

    def getArea(array1):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array1_area = (array1_r - array1_l) * (array1_d - array1_u)
        return array1_area

    def checkArrayUnion(array1, array2):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array2_l = array2[0, 0]
        array2_r = array2[1, 0]
        array2_u = array2[0, 1]
        array2_d = array2[2, 1]
        arrayU_l = max(array1_l, array2_l)
        arrayU_r = min(array1_r, array2_r)
        arrayU_u = max(array1_u, array2_u)
        arrayU_d = min(array1_d, array2_d)
        if arrayU_r > arrayU_l and arrayU_d > arrayU_u:
            arrayU_area = (arrayU_r - arrayU_l) * (arrayU_d - arrayU_u)
            array1_area = (array1_r - array1_l) * (array1_d - array1_u)
            array2_area = (array2_r - array2_l) * (array2_d - array2_u)
            if arrayU_area > array1_area * union_threshold or arrayU_area > array2_area * union_threshold:
                return True
            else:
                return False
        else:
            return False

    def getArrayUnion(array1, array2):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array2_l = array2[0, 0]
        array2_r = array2[1, 0]
        array2_u = array2[0, 1]
        array2_d = array2[2, 1]
        arrayU_l = max(array1_l, array2_l)
        arrayU_r = min(array1_r, array2_r)
        arrayU_u = max(array1_u, array2_u)
        arrayU_d = min(array1_d, array2_d)
        if arrayU_r > arrayU_l and arrayU_d > arrayU_u:
            arrayU_area = (arrayU_r - arrayU_l) * (arrayU_d - arrayU_u)
            return arrayU_area
        else:
            return 0

    # 上下合并box
    new_boxes = []
    vis = [False for _ in range(len(boxes))]
    for i in range(len(boxes)):
        if vis[i]:
            continue
        cur_box = boxes[i]
        flag = True
        while flag:
            flag = False
            for j in range(i + 1, len(boxes)):
                if vis[j]:
                    continue
                if cal_IoU(cur_box, boxes[j]) > row_threshold and check_vertical_dis(cur_box, boxes[j]):
                    vis[j] = True
                    cur_box = mergeArray(cur_box, boxes[j])
                    flag = True
        vis[i] = True
        new_boxes.append(cur_box)
    # 合并交集比例过大的box
    merge_boxes = []
    vis = [False for _ in range(len(new_boxes))]
    for i in range(len(new_boxes)):
        if vis[i]:
            continue
        cur_box = new_boxes[i]
        flag = True
        while flag:
            flag = False
            for j in range(i + 1, len(new_boxes)):
                if vis[j]:
                    continue
                if checkArrayUnion(cur_box, new_boxes[j]):
                    vis[j] = True
                    cur_box = mergeArray(cur_box, new_boxes[j])
                    flag = True
        vis[i] = True
        merge_boxes.append(cur_box)
    # 和其他所有box有交集且比例过大的删除
    final_boxes = []
    cover_area = [0 for _ in range(len(merge_boxes))]
    for i in range(len(merge_boxes)):
        for j in range(i + 1, len(merge_boxes)):
            if i == j:
                continue
            union_area = getArrayUnion(merge_boxes[i], merge_boxes[j])
            cover_area[i] += union_area
            cover_area[j] += union_area
    for i in range(len(merge_boxes)):
        if cover_area[i] <= getArea(merge_boxes[i]) * union_threshold:
            final_boxes.append(merge_boxes[i])
    return final_boxes


# test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000011.jpg')
test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000002.jpg')
# test_char_detect_1('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000002.jpg')
