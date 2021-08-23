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
                draw_1 = cv2.putText(img, str(ibox), (int((x + x_) / 2 - 10), y + 20), font, 1, color=color,
                                     thickness=2)
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
        draw_box(cords_craft_orig, pth_img, pth_img_rect)
        # 在craft画框基础上，再画db框（蓝色）  END
        # draw_box(cords_db_orig, pth_img_rect, pth_img_rect, color=(255, 0, 0))

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
    # 宽度缩放比例， RESIZE_X 参数配置
    RESIZE_X, RESIZE_Y = 0.7, 0.9

    for cord in cords_orig_:  # x方向坐标压缩（避免横向不同大框之间的框交错）
        _cord = xmin, ymin, xmax, ymax = cord[0], cord[1], cord[2], cord[3]
        if RESIZE_X < 1.0:
            _cord = list(scale_x(_cord, RESIZE_X))
            _cord = list(scale_y(_cord, RESIZE_Y))
        cords.append(_cord)

    for cord_cft in cords_craft_orig:
        _cord_cft = int(cord_cft[0]), int(cord_cft[1]), int(cord_cft[4]), int(cord_cft[5])
        if RESIZE_X < 1.0:
            _cord_cft = list(scale_x(_cord_cft, RESIZE_X))
            _cord_cft = list(scale_y(_cord_cft, RESIZE_Y))
        cords_craft.append(_cord_cft)

    # for cord_db in cords_db_orig:
    #     _cord_db = int(cord_db[0]), int(cord_db[1]), int(cord_db[4]), int(cord_db[5])
    #     if RESIZE_X < 1.0:
    #         _cord_db = list(scale_x(_cord_db, RESIZE_X))
    #         _cord_db = list(scale_y(_cord_db, RESIZE_Y))
    #     cords_db.append(_cord_db)

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
        draw_box(_cords_craft, pth_img, pth_img_rect_resize)
        # 在craft画框基础上，再画db框（蓝色）  END
        draw_box(_cords_db, pth_img_rect_resize, pth_img_rect_resize, color=(255, 0, 0))

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
        draw_box(bigboxes, pth_img_rect_resize, pth_img_with_bigbox, color=(0, 128, 0), thickness=2)

    # V. 过滤属于大框内的subbox
    cords.sort(key=lambda pt: pt[0], reverse=True)
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
                 text='0')
        for i, bigitem in bigboxes_subboxes.items():
            bigbox = bigitem['cords_big']
            _sub_boxes = bigitem['sub_boxes']
            # drawbox
            sub_boxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in _sub_boxes]
            color = colors_box[i]
            if i > 0:
                draw_box([bigbox], pth_img_subbox, pth_img_subbox, color=color, thickness=1, text=str(i))
            draw_box(sub_boxes, pth_img_subbox, pth_img_subbox, color=color, thickness=1)
        # END - 重新画大框 和 大框包含的子框 （大框和子框颜色相同，大框编号）

    # VI. 各大框内的subbox融合归并
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
        draw_box([bigboxes_subboxes[0]['cords_big']], pth_img, pth_img_ubox, color=color_ubox, thickness=1, text='0')
        for i, bigitem in bigboxes_subboxes.items():
            bigbox = bigitem['cords_big']
            _uboxes = bigitem['uboxes']
            # drawbox
            uboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in _uboxes]
            color = color_ubox  # colors_box[i]
            if i > 0:
                draw_box([bigbox], pth_img_ubox, pth_img_ubox, color=color, thickness=1, text=str(i))
            draw_box(uboxes, pth_img_ubox, pth_img_ubox, color=color, thickness=1)
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
                _ubox = scale_x(_ubox, resize_x=(10 / 6))
                # y方向缩放10/9
                _ubox = scale_y(_ubox, resize_y=(10 / 9))
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
        draw_box(uboxes_lurd, pth_img, pth_img_uboxes_g, color=color_ubox, thickness=2, seqnum=True)
        draw_box(bigboxes, pth_img_uboxes_g, pth_img_uboxes_g, color=(128, 0, 0), thickness=1, seqnum=True)

    # VIII. 求框中位数，计算字体M, S（双行夹批）

    return res4api_detect_line_union

    # db和craft相交,保留db(例外: craft全包db且远长于db)
    # 重叠矩形(可能是ubox) 判断逻辑


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
    # print(res4api_detect_line)

    # res_detect_line = {
    #     int(itm['name']): {'box': [float(pt) for pt in itm['box']], 'text': itm['text']} for itm in res4api_detect_line
    # }
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

    sorted_res_detect_line = res_detect_line_list
    # print(res_detect_line_list)
    cords = [cord['box'] for cord in sorted_res_detect_line]

    # 以下接入融合算法
    concat_res = boxes_processor(res4api_detect_line, pth_img=pth_img)
    uboxes_g = concat_res['uboxes_g']

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

    widths_line = []
    for cord_ in cords:
        '''
        预备后续非长方形框使用
        '''
        # x1, y1, x2, y2, x3, y3, x4, y4 = cord
        # min_x, max_x = round((x1 + x4) / 2), round((x2 + x3) / 2)
        width = cord_[2] - cord_[0]
        # cord_.append(width)
    w_sorted = sorted(widths_line)
    width_rngs = get_w_rngs(widths_line)
    print(w_sorted)
    print(width_rngs)

    sizes = ['S', 'M', 'L', 'XL', 'XXL', 'XXXL']
    boxgrp_size = {s: [] for s in sizes}
    for i, r_line in enumerate(res4api_detect_line):
        box_line = r_line['box']
        x1, y1, x2, y2, x3, y3, x4, y4 = [int(cord) for cord in box_line]
        min_x, max_x = round((x1 + x4) / 2), round((x2 + x3) / 2)
        min_y, max_y = round((y1 + y2) / 2), round((y3 + y4) / 2)
        width_line = abs(max_x - min_x)

        line_size = get_line_size(width_rngs, width_line)
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
    draw_box(bigboxes, pth_img_with_size, pth_img_with_size, color=(128, 0, 0), thickness=1, seqnum=True)
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
    test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000002.jpg')
    # pth = '/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000018.jpg'
    # test_one(pth)
    # test_one('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/ER007_pure/20_19584_jpg/000024.jpg')
    # test_char_detect_1('/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000002.jpg')
