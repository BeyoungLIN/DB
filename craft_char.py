# -*- coding: utf-8 -*-
# @Time   : 2021/8/15 11:32
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : craft_char.py.py


import os, sys, json, time, traceback

import math, random
import numpy as np
import cv2
import six, base64
from PIL import Image
# from skimage import io

import time
from pprint import pprint
import requests

from tqdm import tqdm
from zhtools.langconv import *

from requests.auth import HTTPBasicAuth

url_line_detect = 'http://api.chinesenlp.com:7001/ocr/v1/line_detect'
url_line_recog = 'http://api.chinesenlp.com:7001/ocr/v1/line_recog'

url_page_recog_0 = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog_0'
url_page_recog = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog'
url_page_recog_1 = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog_1'

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





res4api_detect_line = request_api(url_page_recog_0, param_recog, _auth)

