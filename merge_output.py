# -*- coding: utf-8 -*-
# @Time   : 2021/8/12 15:22
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : merge_output.py.py

# faster-rcnn char detect
def test_char_detect_1(pth_img):
    url_char_detect = '{}/ocr/v1/char_detect_1'.format(api_base_url)
    img = readPILImg(pth_img)

    filename, file_ext = os.path.splitext(os.path.basename(pth_img))
    pth_dir = os.path.abspath( os.path.dirname(pth_img) )
    pth_sav_dir = os.path.join(pth_dir,'output')
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
        int(itm['name']):[float(pt) for pt in itm['box']] for itm in res4api_detect_char
    }
    cords = [v for index,v in res_detect_char.items()]
    # 绘矩形框
    pth_img_rect = os.path.join(pth_sav_dir,filename+'rec.jpg')
    draw_box(cords, pth_img, pth_img_rect)

api_base_url = 'http://api.chinesenlp.com:7001'
AUTH = ('jihe.com','DIY#2020')