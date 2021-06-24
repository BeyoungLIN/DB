# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
from skimage import io
import cv2
import six, base64

from detect.detector0.detector import Detector
from detect.detector2.detector2 import Detector2
from recog.recog0.recog1 import Recog1, empty_img

from detect.detector0.detector import crop_rect
from detect.postdetect import concat_boxes
from postdetect import concat_boxes

blank_img = empty_img


def base64of_img(pth_img):
    image_base64 = ''
    with open(pth_img, 'rb') as f:
        image = f.read()
        image_base64 = str(base64.b64encode(image), encoding='utf-8')
    return image_base64


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


def readPILImg(pth_img):
    img_base64 = base64of_img(pth_img)
    img = base64_to_PIL(img_base64)
    return img


def loadPILImage(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:, :, :3]
    img = np.array(img)

    return Image.fromarray(img)


def crop_img(img, cord):
    image_np = np.array(img)
    # 获取坐标
    _cord = cord[0], cord[1], cord[4], cord[5]
    xmin, ymin, xmax, ymax = _cord
    # 坐标到中心点、高宽转换
    center, size = ((xmin + xmax) / 2, (ymin + ymax) / 2), (xmax - xmin, ymax - ymin)
    rect = center, size, 0
    partImg = crop_rect(image_np, rect)

    return partImg


class RecogHandler():
    def __init__(self):
        print('Initializing detector and recoginer models')
        self.detector = Detector()
        self.detector2 = Detector2()
        self.recog = Recog1()

    def detect_char(self, img, image_path=''):
        '''
        @param img PIL Image
        '''
        res_box_chars = {}
        res4api = []
        if img is not None:
            img = np.array(img)
            res_box_chars = self.detector.detect(img, ocr_type='single_char', image_path=image_path)
        res4api = [
            {
                'box': [str(pt) for pt in box],
                'name': i,
                'text': ''
            } for i, box in res_box_chars.items()
        ]
        return res_box_chars, res4api

    def detect_line(self, img, image_path=''):
        '''
        @param img PIL Image
        '''
        res_box_lines = {}
        res4api = {}
        if img is not None:
            img = np.array(img)
            res_box_lines = self.detector.detect(img, ocr_type='force_colume', image_path=image_path)
        res4api = [
            {
                'box': [str(pt) for pt in box],
                'name': str(i),
                'text': ''
            } for i, box in res_box_lines.items()
        ]
        return res_box_lines, res4api

    def detect_line_1(self, img):
        '''
        @param img PIL Image
        '''
        if img is not None:
            img_np = np.array(img).astype('float32')
            res_box_lines = self.detector2.detect(img_np)
        res4api = [
            {
                'box': [str(pt) for pt in box],
                'name': str(i),
                'text': ''
            } for i, box in res_box_lines.items()
        ]
        return res_box_lines, res4api

    def detect_ocr_line(self, img):
        '''
        @deprecated
        '''
        # dectect line
        res_box_lines, _ = self.detect_line(img)

        # crop line
        PIL_imgs = []
        image_np = np.array(img)
        for i, cord in res_box_lines.items():
            # 获取坐标
            _cord = cord[0], cord[1], cord[4], cord[5]
            xmin, ymin, xmax, ymax = _cord
            # 坐标到中心点、高宽转换
            center, size = ((xmin + xmax) / 2, (ymin + ymax) / 2), (xmax - xmin, ymax - ymin)
            rect = center, size, 0
            partImg = crop_rect(image_np, rect)  # 截取部分
            PIL_imgs.append(partImg)
        res_lines_ocr = []
        for img_line in PIL_imgs:
            res_line_ocr = self.ocr_line(img_line)
            print(res_line_ocr['best'])
            res_lines_ocr.append(res_line_ocr['best'])
        # ocr each line
        res4api = [
            {
                'box': [str(pt) for pt in cord],
                'name': str(i),
                'text': res_lines_ocr[i]['text']
            } for i, cord in res_box_lines.items()
        ]

        return res_box_lines, res_lines_ocr, res4api

    def ocr_line(self, img):
        '''
        @param img PIL Image
        '''
        cands = res_line = self.recog.predict_one(img)
        if len(cands) == 0:
            cands = [{
                'confidence': 0.0,
                'char': ''
            }]
        res = {
            'best': {
                'confidence': cands[0]['confidence'],
                'text': cands[0]['char']
            },
            'cands': [
                {
                    'confidence': cand['confidence'],
                    'text': cand['char']
                } for cand in cands
            ]
        }
        return res

    def recog_line(self, img):
        '''
        @ recog_line升级为recog_page现在功能,
        @ recog_page升级为检测识别列/检测识别列中的字
        @param img PIL Image
        '''

        cands = res_line = self.recog.predict_one(img)
        if len(cands) == 0:
            cands = [{
                'confidence': 0.0,
                'char': ''
            }]
        res = {
            'best': {
                'confidence': cands[0]['confidence'],
                'text': cands[0]['char']
            },
            'cands': [
                {
                    'confidence': cand['confidence'],
                    'text': cand['char']
                } for cand in cands
            ]
        }
        return res

    def ocr_many(self, imgList):
        res = self.recog.predict(imgList)
        return res

    def rng_interact(self, rng1, rng2):
        interacted = False
        xmin1, xmax1 = rng1[0], rng1[1]
        xmin2, xmax2 = rng2[0], rng2[1]
        if xmin1 <= xmax2 and xmin2 <= xmax1: interacted = True
        rng_u = [min(xmin1, xmin2), max(xmax1, xmax2)]
        return interacted, rng_u

    def get_w_rngs(self, widths, R=0.1):
        w_sorted = sorted(widths)
        w_rngs_tmp = [[w * (1 - R), w * (1 + R)] for w in w_sorted]
        w_rngs, w_rng = [], w_rngs_tmp.pop(0)
        for _rng in w_rngs_tmp:
            interacted, rng_u = self.rng_interact(w_rng, _rng)
        if interacted:
            w_rng = rng_u
        else:
            w_rngs.append(w_rng)
            w_rng = _rng
        return w_rngs

    def get_line_size(self, w_rngs, w):
        sizes = ['S', 'M', 'L', 'XL', 'XXL', 'XXXL']
        for i, rng in enumerate(w_rngs):
            rngl, rngr = rng[0], rng[1]
            if rngl <= w <= rngr: return sizes[i]
        return sizes[-1]

    def ocr_page(self, img, detector='craft'):
        adv_res, concat_res = {}, {}
        # 默认是craft行检测
        res_detect_line, res4api_detect_line = self.detect_line(img)
        if 'craft_char' == detector:
            res_detect_line, res4api_detect_line = self.detect_char(img)
        if 'db' == detector:
            # 替换成db的行检测
            res_detect_line, res4api_detect_line = self.detect_line_1(img)
        if 'mix' == detector:
            # 同时求db的行检测，并进行融合, 返回简单结果格式
            res_detect_line, res4api_detect_line = self.detect_line(img)
            res_detect_line_db, res4api_detect_line_db = self.detect_line_1(img)
            concat_res = concat_boxes(res4api_detect_line, res4api_detect_line_db)
        if 'adv' == detector:
            # 同时求db的行检测，并进行融合, 返回复杂结果格式
            res_detect_line, res4api_detect_line = self.detect_line(img)
            res_detect_line_db, res4api_detect_line_db = self.detect_line_1(img)
            concat_res = concat_boxes(res4api_detect_line, res4api_detect_line_db)
        img_lst = []

        if 'mix' == detector or 'adv' == detector:
            uboxes_g = concat_res['uboxes_g']
            res_detect_line = {i: [ub[0], ub[1], ub[2], ub[1], ub[2], ub[3], ub[0], ub[3]] \
                               for i, ub in enumerate(uboxes_g)}
            res4api_detect_line = [
                {
                    'box': [str(pt) for pt in box],
                    'name': str(i),
                    'text': ''
                } for i, box in res_detect_line.items()
            ]

        widths_line = []
        for index, cord in res_detect_line.items():
            try:
                img_line = crop_img(img, cord)
                img_lst.append(img_line)

                x1, y1, x2, y2, x3, y3, x4, y4 = cord
                min_x, max_x = round((x1 + x4) / 2), round((x2 + x3) / 2)
                widths_line.append(abs(max_x - min_x))
            except Exception as e:
                print(e)
                continue

        width_rngs = self.get_w_rngs(widths_line)
        res_ocr_many = self.ocr_many(img_lst)

        for i, r_line in enumerate(res4api_detect_line):
            res_ocr_line = res_ocr_many[i]
            cands = res_ocr_line['cands']
            cand = cands[0]['char']
            r_line['text'] = cand
            # 字检测结果
            len_chars = len(cand)  # 字数
            box_line = r_line['box']
            x1, y1, x2, y2, x3, y3, x4, y4 = box_line = [int(cord) for cord in box_line]
            min_x, max_x = round((x1 + x4) / 2), round((x2 + x3) / 2)
            min_y, max_y = round((y1 + y2) / 2), round((y3 + y4) / 2)
            width_line = abs(max_x - min_x)

            boxes_char = []
            if len_chars > 0:
                h_char = (max_y - min_y) / len_chars
                for j in range(len_chars):
                    box = [
                        min_x, min_y + j * h_char, max_x, min_y + j * h_char,
                        max_x, min_y + (j + 1) * h_char, min_x, min_y + (j + 1) * h_char
                    ]
                    box_char = {
                        'box': [str(round(pt)) for pt in box],
                        'name': j,
                        'text': cand[j]
                    }
                    boxes_char.append(box_char)
            r_line['boxes_char'] = boxes_char
            line_size = self.get_line_size(width_rngs, width_line)
            # r_line['size'] = 'M'
            r_line['size'] = line_size

        res = res4api_detect_line
        if 'adv' == detector:
            res = adv_res = {
                'res_basic': res4api_detect_line,
                'big_sub_boxes': concat_res['bigboxes_uboxes']
            }
        return res


craft_db = RecogHandler()#实例化类
# print(my_computer.screen)#打印类中的属性值
craft_db.ocr_page('/Users/Beyoung/Desktop/Projects/AC_OCR/OCR测试图像2/史记1.jpg','mix')#启动类中的方法
# ocr_page()

