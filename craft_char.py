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
# from skimage import io
from typing import List

import cv2
import numpy as np
import requests
import six
from PIL import Image
from sklearn.cluster import DBSCAN, MeanShift, OPTICS, Birch

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
                draw_1 = cv2.putText(img, str(ibox), (int((x + x_) / 2 - 10), y + 20), font, 1, color=color, thickness=2)
            if not '' == text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                draw_1 = cv2.putText(img, text, (int((x + x_) / 2 - 10), y + 16), font, 1, color=color, thickness=2)

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
    res4api_detect_line = request_api(url_page_recog_1, param_recog, _auth)
    # res4api_detect_line = request_api(url_page_recog, param_recog, _auth)
    # print(res4api_detect_line)

    res_detect_line = {
        int(itm['name']): {'box': [float(pt) for pt in itm['box']], 'text': itm['text']} for itm in res4api_detect_line
    }
    res_detect_line_list = [
        {'box': [float(pt) for pt in itm['box']], 'text': itm['text']} for itm in res4api_detect_line
    ]
    out_sav = ''
    cords = [v['box'] for index, v in res_detect_line.items()]
    cords_np = []
    # print(cords)

    sorted_res_detect_line = bubble_sort(res_detect_line_list)
    # sorted_res_detect_line = res_detect_line_list
    # print(res_detect_line_list)
    cords = [cord['box'] for cord in sorted_res_detect_line]
    pth_img_rect = os.path.join(pth_sav_dir, filename + 'rec_db.jpg')
    draw_box(cords, pth_img, pth_img_rect, seqnum=True, resize_x=0.8,show_img=True)

    # for cord in cords:
    #     cord = np.array(cord)
    #     cord = cord.reshape(-1, 2)
    #     cords_np.append(cord)
    # # cords = adjustColumeBoxes(cords)
    # # cords = adjustBoxesoutput(cords_np)
    # boxes = []
    # print(cords_np)
    # boxes_np = cluster_sort(cords_np)
    #
    # for box in boxes_np:
    #     box = box.reshape(-1, 8)
    #     for box_ in box.tolist():
    #         print(box_)
    #         # for box__ in box_:
    #         boxes.append(box_)
    # print(boxes)
    #
    #
    # pth_img_rect = os.path.join(pth_sav_dir, filename + 'rec.jpg')
    # draw_box(boxes, pth_img, pth_img_rect, seqnum=True, resize_x=0.8,show_img=True)
    res_txt = ''
    # print(res_detect_line.get('text'))
    for box in sorted_res_detect_line:
        # print(box.get('text'))
        res_txt += box['text'] + '\n'
    # res_txt = [box['text'] + '\n' for box in sorted_res_detect_line]
    print(res_txt)
    # with open(os.path.join(pth_sav_dir, filename) + '.txt', 'w') as f:
        # f.write(res_txt)


def bubble_sort(items):
    for i in range(len(items) - 1):
        flag = False
        for j in range(len(items) - 1 - i):
            if not check_order(items[j], items[j + 1]):
                # if not check_order(items.get[j], items.get[j + 1]):
                #     items.get[j], items.get[j + 1] = items.get[j + 1], items.get[j]
                items[j], items[j + 1] = items[j + 1], items[j]
                flag = True
        if not flag:
            break
    return items


def check_order(dict_1, dict_2):
    '''

    :param list_1:
    :param list_2:
    :return:
    True means list1 is before list2, False means list2 is before list 1
    '''
    if dict_1.get('box')[0] >= dict_2.get('box')[2]:
        return True
    elif dict_1.get('box')[1] < dict_2.get('box')[3]:
        return True
    else:
        return False


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


# test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000011.jpg')
# test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000002.jpg')
test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000174.jpg')
# test_char_detect_1('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000002.jpg')
