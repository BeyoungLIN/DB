# coding:utf-8
import os,sys,json,time,traceback

import math,random
import numpy as np
import cv2
import six,base64
from PIL import Image
# from skimage import io

import time
from pprint import pprint
import requests

from tqdm import tqdm
from zhtools.langconv import *

from requests.auth import HTTPBasicAuth

''' '''
# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

# 转换简体到繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line


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

def crop_rect(img, rect ,alph = 0.15):
    img  = np.asarray(img)
    # get the parameter of the small rectangle
    # # print("rect!")
    # # print(rect)
    center, size, angle = rect[0], rect[1], rect[2]
    min_size  = min(size)

    if(angle>-45):
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # angle-=270
        size  = ( int(size[0] + min_size*alph ) , int(size[1]  +  min_size*alph) )
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
    # size = tuple([int(rect[1][1]), int(rect[1][0])])
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite("debug_im/img_rot.jpg", img_rot)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center=tuple(map(int,center))
        size = tuple([int(rect[1][1]), int(rect[1][0])])
        size  = ( int(size[0] + min_size*alph) ,int(size[1]  + min_size*alph) )
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite("debug_im/img_rot.jpg", img_rot)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    img_crop = Image.fromarray(img_crop)
    return img_crop

def crop_img(img, cord):
    image_np = np.array(img)
    # 获取坐标
    _cord = cord[0], cord[1], cord[4], cord[5]
    xmin,ymin,xmax,ymax = _cord 
    # 坐标到中心点、高宽转换
    center, size = ((xmin+xmax)/2,(ymin+ymax)/2), (xmax-xmin,ymax-ymin)
    rect = center, size, 0
    partImg = crop_rect(image_np, rect)
    
    return partImg

def cv_imread(file_path = ""):
        img_mat=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        return img_mat

def cv_imwrite(file_path , frame ): 
    cv2.imencode('.jpg', frame)[1].tofile(file_path)



def draw_box(cords, pth_img,pth_img_rect, color=(0,0,255), resize_x=1.0,thickness=1, text='', seqnum=False):
    try:
        # img = cv2.imread(pth_img)
        img = cv_imread(pth_img)  # 解决中文路径文件的读
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        boxes = []

        for cord in cords:
            _cord = int(cord[0]), int(cord[1]), int(cord[4]), int(cord[5])
            xmin,ymin,xmax,ymax = _cord
            if resize_x < 1.0:
                _cord = scale_x(_cord, resize_x)
            boxes.append(_cord)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for ibox, box in enumerate(boxes):
            x, y, x_, y_ = box
            # draw_1 = cv2.rectangle(img, (x,y), (x_,y_), (0,0,255), 2 )
            draw_1 = cv2.rectangle(img, (x,y), (x_,y_), color, thickness)
            if seqnum:
                draw_1 = cv2.putText(img, str(ibox), (int((x+x_)/2-10), y+20), font, 0.6,color=color)
            if not ''==text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                draw_1 = cv2.putText(img, text, (int((x+x_)/2-10), y+16), font, 0.4,color=color)
        
        # print('Writing to image with rectangle {}\n'.format(pth_img_rect))

        cv_imwrite(pth_img_rect, draw_1)  # 解决中文路径文件的写
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, e.__str__(), ',File', fname, ',line ',exc_tb.tb_lineno)
        traceback.print_exc()  # debug.error(e)

def draw_line(cords, pth_img,pth_img_rect, color=(0,0,255), thickness=2):
    try:
        # img = cv2.imread(pth_img)
        img = cv_imread(pth_img)  # 解决中文路径文件的读
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lines = []

        for cord in cords:
            _cord = int(cord[0]), int(cord[1]), int(cord[2]), int(cord[3])
            xmin,ymin,xmax,ymax = _cord
            lines.append(_cord)

        for line in lines:
            x, y, x_, y_ = line
            
            draw_1 = cv2.line(img, (x,y), (x_,y_), color, thickness )
        
        # print('Writing to image with rectangle {}\n'.format(pth_img_rect))

        cv_imwrite(pth_img_rect, draw_1)  # 解决中文路径文件的写
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, e.__str__(), ',File', fname, ',line ',exc_tb.tb_lineno)
        traceback.print_exc()  # debug.error(e)

def draw_line_inbox(lst_cords, pth_img_rect, pth_img_char_split):
    '''
    在行内画字间隔线
    '''
    try:
        img = img = cv_imread(pth_img_rect)  # 解决中文路径文件的读
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        boxes = []
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, e.__str__(), ',File', fname, ',line ',exc_tb.tb_lineno)
        traceback.print_exc()  # debug.error(e)

def test_improc_api(pth_img, pth_out):
    url_line_improc = 'http://api.chinesenlp.com:7001/ocr/v1/pic_preproc'
    # pth_img = "E:/Projs/AncientBooks/src/data/1.jpg"
    # pth_out = "E:/Projs/AncientBooks/src/data/1.out.jpg"
    picstr = base64of_img(pth_img)
    param_improc = {
        'picstr': picstr,
        'mod_proc': 'contrast'
    }
    r=requests.post(url_line_improc, param_improc,auth=('jihe.com','DIY#2020'))
    o = json.loads(r.text)
    # print(o['data'].keys())
    print(o['msg'])
    b64 = o['data']['contrasted']
    img = base64_to_PIL(b64)
    img.save(pth_out)

def test_improc_api_bydir(pth_dir):
    dir_test_img = pth_dir
    lst_pthimg = [os.path.join(dir_test_img, pic) for pic in os.listdir(dir_test_img) ] 
    for pthimg in tqdm(lst_pthimg):
        fname_img = os.path.basename(pthimg)
        name, ext = os.path.splitext(fname_img)
        pthout = os.path.join(pth_dir, '{}_c{}'.format(name, ext))
        print('preprocessing {}...and saving {}'.format(pthimg, pthout))
        try:
            test_improc_api(pthimg, pthout)
        except Exception as e:
            print(e)

url_line_detect = 'http://api.chinesenlp.com:7001/ocr/v1/line_detect'
url_line_recog = 'http://api.chinesenlp.com:7001/ocr/v1/line_recog'

url_page_recog_0 = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog_0'
url_page_recog = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog'
url_page_recog_1 = 'http://api.chinesenlp.com:7001/ocr/v1/page_recog_1'
# url_line_detect = 'http://192.168.10.47:1688/ocr/v1/line_detect'

def request_api(url_api, params, _auth):
    r = requests.post(url_api, data=params, auth=_auth)
    str_res = r.text
    o_res = json.loads(str_res)
    res = o_res['data']
    return res

def test_one(pth_img):
    img = readPILImg(pth_img)

    filename, file_ext = os.path.splitext(os.path.basename(pth_img))
    pth_dir = os.path.abspath( os.path.dirname(pth_img) )
    pth_sav_dir = os.path.join(pth_dir,'output' )

    if not os.path.exists(pth_sav_dir):
        os.makedirs(pth_sav_dir)

    pth_sav = os.path.join(pth_sav_dir,filename+'_res_recog.txt')
    save_folder = os.path.join(pth_sav_dir,filename)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(pth_sav):
        print('{} already exists'.format(pth_sav))
        return

    imgbase64 = base64of_img(pth_img)
    
    # param_detect = {
    #     'picstr': imgbase64,
    #     'mod': 'base64',
    #     'do_ocr': 0
    # }``
    # r = requests.post(url_line_detect, data=param_detect)
    _auth = ('jihe.com','DIY#2020')
    param_recog = {
        'picstr': imgbase64,
        'mod': 'base64'
    }

    '''
    # craft_char检测与识别
    res4api_detect_line = request_api(url_page_recog_0, param_recog, _auth)

    '''
    # craft检测与识别
    res4api_detect_line = request_api(url_page_recog, param_recog, _auth)
    # print(res4api_detect_line)
    # db检测与识别
    res4api_detect_line_db = request_api(url_page_recog_1, param_recog, _auth)
    
    # 画db框（红色）  BEGIN  check:这里红色框应该是craft
    res_detect_line = {
        int(itm['name']):{'box':[float(pt) for pt in itm['box']], 'text': itm['text'] } for itm in res4api_detect_line
    }
    out_sav = ''
    cords = [v['box'] for index,v in res_detect_line.items()]

    pth_img_rect = os.path.join(pth_sav_dir,filename+'rec.jpg')
    draw_box(cords, pth_img, pth_img_rect, resize_x=0.8 )
    # 画db框（红色）  END
    ''' ''' 
    # 在craft画框基础上，再画db框（蓝色）  BEGIN
    res_detect_line_db = {
        int(itm['name']):{'box':[float(pt) for pt in itm['box']], 'text': itm['text'] } for itm in res4api_detect_line_db
    }
    cords_db = [v['box'] for index,v in res_detect_line_db.items()]
    draw_box(cords_db, pth_img_rect, pth_img_rect, color=(255,0,0), resize_x=0.8)
    # 在craft画框基础上，再画db框（蓝色）  END
    
    # 存craft结果json
    pth_json_res = os.path.join(pth_sav_dir, filename+'.json.txt' )
    with open(pth_json_res, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(res4api_detect_line, indent=2, ensure_ascii=False))
    
    ''' '''
    # 存db结果json
    pth_json_res_db = os.path.join(pth_sav_dir, filename+'_db.json.txt' )
    with open(pth_json_res_db, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(res4api_detect_line_db, indent=2, ensure_ascii=False))
    


    ''' '''
    # pack results of  ocr_line
    for index,v in res_detect_line.items():
        # print(v)
        try:
            img_line = crop_img(img, v['box'])
            # print()
            pth_img_sav = save_folder+"/col_{}.jpg".format(index)
            # 保存截取的图片
            partImg_array  = np.uint8(img_line)
            partImg = Image.fromarray(partImg_array).convert("RGB")
            partImg.save(pth_img_sav)

            pth_col = pth_img_sav
            picstr= base64of_img(pth_col)
            param_recog = {
                'picstr': picstr
            }
            # _auth = ('jihe.com', 'DIY#2020')
            r = requests.post(url_line_recog, data=param_recog, auth=('jihe.com','DIY#2020'))
            print(r)
            str_res = r.text
            o_res = json.loads(str_res)
            res_ocr_line = o_res['data']

            pprint(res_ocr_line)
            res_line = res_ocr_line['best']['text']

            res_line = v['text']
            out_sav += (res_line+'\n')
            pth_recog_sav = save_folder+"/col_{}_recog.txt".format(index)
                            
            
            with open(pth_recog_sav, 'w+', encoding='utf-8') as f:
                f.write(res_line)
                # print(res_line)
        except Exception as e:
            print(e)
            continue
    with open(pth_sav,'w+',encoding='utf-8') as f:
        f.write(out_sav.strip())
    
    pth_sav_simplified = os.path.join(pth_sav_dir,filename+'_res_recog.simplified.txt')
    out_sav_sim = '\n'.join([cht_to_chs(ln) for ln in out_sav.split('\n')])
    with open(pth_sav_simplified,'w+',encoding='utf-8') as f:
        f.write(out_sav_sim.strip())

def test_by_dir(pth_dir):
    t1 = time.time()
    print('Begin Testing.')
    dir_test_img = pth_dir
    lst_pthimg = [os.path.join(dir_test_img, pic) for pic in os.listdir(dir_test_img)]   #  if pic.endswith('_c.tif')

    ts = []
    for pthimg in tqdm(lst_pthimg):
        try:
            print('Recognizing {}...'.format(pthimg))
            _t1 = time.time()

            test_one(pthimg)

            _t2 = time.time()
            ts.append(_t2-_t1)
        except Exception as e:
            print(e)
            print(pthimg)
            continue
    
    print(ts)
    print('End Testing.')
    t2 = time.time()
    print('Time comsuption {} seconds, average {} seconds/per pic.'.format((t2-t1), (t2-t1)/len(lst_pthimg)))

def test_char_detect_1(pth_img):
    url_char_detect = 'http://api.chinesenlp.com:7001/ocr/v1/char_detect_1'
    img = readPILImg(pth_img)

    filename, file_ext = os.path.splitext(os.path.basename(pth_img))
    pth_dir = os.path.abspath( os.path.dirname(pth_img) )
    pth_sav_dir = os.path.join(pth_dir,'output' )

    if not os.path.exists(pth_sav_dir):
        os.makedirs(pth_sav_dir)

    imgbase64 = base64of_img(pth_img)
    param_detect = {
        'picstr': imgbase64,
        'mod': 'base64',
        'do_ocr': 0
    }
    r = requests.post(url_char_detect, data=param_detect)
    str_res = r.text
    o_res = json.loads(str_res)
    res4api_detect_char = o_res['data']
    res_detect_char = {
        int(itm['name']):[float(pt) for pt in itm['box']] for itm in res4api_detect_char
    }
    cords = [v for index,v in res_detect_char.items()]

    pth_img_rect = os.path.join(pth_sav_dir,filename+'rec.jpg')
    draw_box(cords, pth_img, pth_img_rect)

def scale_x(_cords, resize_x):
    xmin,ymin,xmax,ymax = _cords
    xmid = (xmin+xmax)/2
    xmin_ = math.ceil(xmid - resize_x*(xmid-xmin))
    xmax_ = math.floor(xmid + resize_x*(xmax-xmid))

    return xmin_, ymin, xmax_, ymax

def scale_y(_cords, resize_y):
    xmin,ymin,xmax,ymax = _cords
    ymid = (ymin+ymax)/2
    ymin_ = math.ceil(ymid - resize_y*(ymid-ymin))
    ymax_ = math.floor(ymid + resize_y*(ymax-ymid))

    return xmin, ymin_, xmax, ymax_

from colors import COLORS
def randcolors(n=10):
    _colors = []
    for color_series, color_dct in COLORS.items():
        for k, v_rgb in color_dct.items():
            _colors.append(tuple([ int(num) for num in v_rgb.split(',')]))
    # return [ random.choice(_colors) for i in range(n)]
    return random.sample(_colors, n)

from postdetect import concat_boxes
def ajust_boxes(pth_img, dbg=False):
    test_one(pth_img)

    filename, file_ext = os.path.splitext(os.path.basename(pth_img))
    pth_dir = os.path.abspath( os.path.dirname(pth_img) )
    pth_sav_dir = os.path.join(pth_dir,'output' )
    # pth_sav_dir = os.path.join(pth_dir )

    # 〇.分别获取craft和db的api结果
    # # craft检测与识别结果json
    pth_json_res = os.path.join(pth_sav_dir, filename+'.json.txt' )
    with open(pth_json_res, 'r', encoding='utf-8') as f:
        c = f.read()
        res4api_detect_line = json.loads(c)
        print(res4api_detect_line)
    ''' '''
    # db检测与识别结果json
    pth_json_res_db = os.path.join(pth_sav_dir, filename+'_db.json.txt' )
    with open(pth_json_res_db, 'r', encoding='utf-8') as f:
        c1=f.read()
        res4api_detect_line_db = json.loads(c1)
        # print(res4api_detect_line_db)
    
    # concat_boxes(res4api_detect_line, res4api_detect_line_db, pth_img=pth_img, dbg=dbg)
    res_mix = concat_boxes(res4api_detect_line, res4api_detect_line_db, pth_img=pth_img, dbg=dbg)
    print(res_mix)
    # 此时, res_mix = res4api_detect_line_union
    return res_mix

    # uboxes_g 切图后整体识别，重组结构（ubox_g， bigbox的识别文本存储）

def main():
    '''
    E:\Projs\AncientBooks\src\data\test2\siku
    E:\Projs\AncientBooks\src\data\test2\205（四库全书）
    E:\Projs\AncientBooks\src\data\test2\206（史记）
    '''
    pth_dir = sys.argv[1]
    # test_improc_api_bydir(pth_dir)
    test_by_dir(pth_dir)

    # pth_line = r'E:\Projs\AncientBooks\src\data\test2\minguo\0\output\col_17.jpg'
    # test_char_detect_1(pth_line)

def main1(pth_img):
    # pth_img = r'E:\Projs\AncientBooks\src\test\api_set\page\0\image_002_看图王.jpg'
    # 用新db测试一遍 史记9
    # pth_img = r'/Users/Beyoung/Desktop/Projects/DB/datasets/new_test/图片 1.png'
    # pth_img = r'E:\Projs\AncientBooks\src\test\api_set\page\0\ZHSY003341-000009.tif'
    # pth_img = r'E:\Projs\AncientBooks\src\test\api_set\page\0\绝妙好词笺.七卷.宋.周密原辑.清.查为仁.厉鹗同笺.清乾隆十五年宛平查氏澹宜书屋刊本 (1) - 0006(1).tif'
    ajust_boxes(pth_img, dbg=True)

if __name__=="__main__":
    # main()
    # test_improc_api('','')
    # main1(r'/Users/Beyoung/Desktop/Projects/AC_OCR/金陵诗徵/24561622248357_.pic_hd.jpg')
    pth_img = '/Users/Beyoung/Desktop/Projects/AC_OCR/OCR测试图像2/史记1.jpg'
    res4api_detect_line_union = ajust_boxes(pth_img, dbg=False)
    # print(res4api_detect_line_union)
    # res_detect_line_union = {
    #     int(itm['name']): {'box': [float(pt) for pt in itm['box']], 'text': itm['text']} for itm in res4api_detect_line_union
    # }
    # # cords_db = [v['box'] for index, v in res_detect_line_union.items()]
    # # print(ajust_boxes('/Users/Beyoung/Desktop/Projects/AC_OCR/OCR测试图像2/史记1.jpg', dbg=False))
    # # pack results of  ocr_line
    #
    # img = readPILImg
    #
    # filename, file_ext = os.path.splitext(os.path.basename(pth_img))
    # pth_dir = os.path.abspath( os.path.dirname(pth_img) )
    # pth_sav_dir = os.path.join(pth_dir,'output' )
    #
    # if not os.path.exists(pth_sav_dir):
    #     os.makedirs(pth_sav_dir)
    #
    # pth_sav = os.path.join(pth_sav_dir,filename+'_res_recog.txt')
    # save_folder = os.path.join(pth_sav_dir,filename)
    #
    # for index, v in res_detect_line_union.items():
    #     try:
    #         img_line = crop_img(img, v['box'])
    #         pth_img_sav = save_folder + "/col_{}.jpg".format(index)
    #         # 保存截取的图片
    #         partImg_array = np.uint8(img_line)
    #         partImg = Image.fromarray(partImg_array).convert("RGB")
    #         partImg.save(pth_img_sav)
    #
    #         pth_col = pth_img_sav
    #         picstr= base64of_img(pth_col)
    #         param_recog = {
    #             'picstr': picstr
    #         }
    #         r = requests.post(url_line_recog, data=param_recog)
    #         print(r)
    #         str_res = r.text
    #         o_res = json.loads(str_res)
    #         res_ocr_line = o_res['data']
    #
    #         pprint(res_ocr_line)
    #         res_line = res_ocr_line['best']['text']
    #
    #         res_line = v['text']
    #         out_sav += (res_line + '\n')
    #         pth_recog_sav = save_folder + "/col_{}_recog.txt".format(index)
    #
    #         with open(pth_recog_sav, 'w+', encoding='utf-8') as f:
    #             f.write(res_line)
    #     except Exception as e:
    #         print(e)
    #         continue
    # with open(pth_sav, 'w+', encoding='utf-8') as f:
    #     f.write(out_sav.strip())