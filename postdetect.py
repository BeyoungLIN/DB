#coding:utf-8


import os,sys,json,time,traceback

import math,random
import numpy as np
import cv2
import six,base64
from PIL import Image
# from skimage import io

import time
from pprint import pprint

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

'''
post detecting functions begin here
'''

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

def conv_cords(cord):
    # 将[ 左上，右上，右下，左下 ] 的坐标 转换为 [左上，右下，宽，长，宽长比] 格式
    _cord = xmin,ymin,xmax,ymax = int(cord[0]), int(cord[1]), int(cord[4]), int(cord[5])
    w = xmax - xmin
    h = ymax - ymin
    w_h = w/h
    return list(_cord) + [w, h, w_h]

def line_interact(line1, line2, minDot, axis='x'):
    Xmin, Ymin = minDot
    # 判断两线段是否相叠
    interacted, interacted_x, interacted_y = False, False, False
    line_u=[]
    xmin1, xmax1, ymin1, ymax1 = line1[0], line1[2], line1[1], line1[3]
    xmin2, xmax2, ymin2, ymax2 = line2[0], line2[2], line2[1], line2[3]
    _xmin = min(xmin1, xmin2)
    # 相叠判断逻辑
    if xmin1 <= xmax2 and xmin2 <= xmax1: interacted_x = True
    if ymin1 <= ymax2 and ymin2 <= ymax1: interacted_y = True
    # 线段并
    line_u_x = [ min(xmin1, xmin2), Ymin-2, max(xmax1, xmax2), Ymin-2 ]
    line_u_y = [ _xmin-2, min(ymin1, ymin2), _xmin-2, max(ymax1, ymax2) ]
    # 线段交
    line_inter_x = [ max(xmin1, xmin2), Ymin-2, min(xmax1, xmax2), Ymin-2]
    line_inter_y = [ Xmin-2, max(ymin1, ymin2), Xmin-2 , min(ymax1, ymax2)]

    interacted, line_u, line_inter = interacted_x, line_u_x, line_inter_x
    if 'y'==axis:
        interacted, line_u, line_inter = interacted_y, line_u_y, line_inter_y
    return interacted, line_u, line_inter

def rec_interact(box1, box2, minDot):
    xmin1, ymin1, xmax1, ymax1 = box1[0], box1[1], box1[2], box1[3]
    xmin2, ymin2, xmax2, ymax2 = box2[0], box2[1], box2[2], box2[3]
    line1x = [ xmin1,ymin1, xmax1,ymin1 ]
    line2x = [ xmin2,ymin2, xmax2,ymin2 ]
    line1y = [ xmin1,ymin1, xmin1,ymax1 ]
    line2y = [ xmin2,ymin2, xmin2,ymax2 ]
    interacted_x, interacted_y = False, False
    interacted_x, linex_u, linex_inter = line_interact(line1x, line2x, minDot)
    interacted_y, liney_u, liney_inter = line_interact(line1y, line2y, minDot, axis='y')
    return (interacted_x and interacted_y), linex_inter, liney_inter

def filter_subbox(cords, bigbox, minDot):
    cord_ = [bigbox[0], bigbox[1], bigbox[4], bigbox[5]]
    subboxes = [
        cord for cord in cords \
        if rec_interact(cord, cord_, minDot)[0]
    ]
    return subboxes

def union_2box(box1, box2, cords_db):
    #合并两个相交的box
    xmin1, ymin1, xmax1, ymax1 = box1[0], box1[1], box1[2], box1[3]
    xmin2, ymin2, xmax2, ymax2 = box2[0], box2[1], box2[2], box2[3]
    xmin, ymin = min(xmin1, xmin2), min(ymin1, ymin2)
    xmax, ymax = max(xmax1, xmax2), max(ymax1, ymax2)
    
    '''  
    # 判断是否只有cord_craft成员
    if box1 in cords_db and not box2 in cords_db:
        _xmin,_ymin, _xmax,_ymax = xmin1, ymin1, xmax1, ymax1
        # if ymax-ymin > _ymax-_ymin: _ymax, _ymin = ymax, ymin
        # if _xmin > xmin and _xmax < xmax: _xmin, _xman = xmin, xmax
        return _xmin,_ymin, _xmax,_ymax
    if not box1 in cords_db and box2 in cords_db:
        _xmin,_ymin, _xmax,_ymax = xmin2, ymin2, xmax2, ymax2
        # if ymax-ymin > _ymax-_ymin: _ymax, _ymin = ymax, ymin
        # if _xmin > xmin and _xmax < xmax: _xmin, _xman = xmin, xmax
        return _xmin,_ymin, _xmax,_ymax
'''
    if box2 in cords_db:
        ymin, ymax = ymin2, ymax2
        if ymin1<ymin and ymax1>ymax:
            ymin, ymax = ymin1, ymax1
    if not (box2 in cords_db)  and not ( ymin2<ymin1 and ymax2>ymax1 ):
        ymin, ymax = ymin1, ymax1
    return [xmin,ymin, xmax,ymax]

def union_subboxes(sub_boxes, cords_db, minDot, URatio=0.6):
    # print(sub_boxes)
    sub_boxes.sort(key=lambda pt:pt[1])
    # print(sub_boxes)
    # pth_img = '/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/000174.jpg'
    # pth_img_rect = '/Users/Beyoung/Desktop/Projects/ER/dataset/ER007/20_19584_jpg/output/000174rec.jpg'
    # _sub_boxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in sub_boxes]
    # pth_img_ubox = pth_img_rect.replace('rec.jpg', 'rec_ubox_after_sort.jpg')
    # draw_box(_sub_boxes, pth_img, pth_img_ubox, color=(0, 128, 0), seqnum=True, thickness=1,text='0')
    boxes_union = []
    while len(sub_boxes)>0:
        # box_u = sub_boxes[0]
        # sub_boxes.remove(box_u)
        box_u = sub_boxes.pop(0)
        inter_u = []
        for i, sub_box in enumerate(sub_boxes):
            interacted, linex_inter, liney_inter = rec_interact(box_u, sub_box, minDot)
            # 相交， x和y方向相交率 > 0.6  (0.6这个值需要测)
            lx_boxu, ly_boxu = abs(box_u[2]-box_u[0]), abs(box_u[3]-box_u[1])
            lx_subbox, ly_subbox = abs(sub_box[2]-sub_box[0]), abs(sub_box[3]-sub_box[1])
            lx_inter, ly_inter = abs(linex_inter[2]-linex_inter[0]), abs(liney_inter[3]-liney_inter[1])

            if interacted:
                # print('ly_inter/ly_boxu:\t{}'.format(ly_inter/ly_boxu))
                # if not sub_box in cords_db:
                inter_u.append(sub_box)
                if (ly_inter/ly_boxu) < 0.08:
                    continue
                if ly_subbox < lx_subbox and \
                   ((lx_inter/lx_subbox) < URatio or (lx_inter/lx_boxu) < URatio ): 
                    continue
                box_u = union_2box(box_u, sub_box, cords_db)
                # 相交， x和y方向有一个方向 < 0.6 
                # < 0.6的那个方向，不参与合并，但参与remove
        for box_ in inter_u: sub_boxes.remove(box_)
        boxes_union.append(box_u)
    return boxes_union

# y轴方向线段归并
def liney_union(lines, minDot):
    if len(lines) <=1: return lines
    lines_union, line_u = [], lines[0]

    for i in range(1,len(lines)):
        _line = lines[i]
        interacted, _line_u, _ = line_interact(line_u, _line, minDot, axis='y')
        if interacted:
            line_u = _line_u
        else:
            lines_union.append(line_u)
            line_u = _line
    lines_union.append(line_u)
    return lines_union

def line_inter_box(line, box, minDot, axis='x'):
    _xmin, _ymin, _xmax, _ymax = line[0], line[1], line[2], line[3]
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    liney_box = [xmin, ymin, xmin, ymax]
    linex_box = [xmin, ymin, xmax, ymin]

    interacted = False
    interacted_x, line_u_x, _ix = line_interact(line, liney_box, minDot)
    interacted_y, line_u_y, _iy = line_interact(line, liney_box, minDot, axis='y')
    if 'y'==axis: interacted = interacted_y
    return interacted


def ygroup_uboxes(_uboxes, minDot):
    ygrp_uboxes, ulines_y = {}, []
    _cords = _uboxes
    # 7.0 竖向坐标缩放0.9
    # 高度缩放比例， RESIZE_Y 参数配置
    # RESIZE_Y = 0.9
    # _cords = [ list(scale_y(_cord, RESIZE_Y)) for _cord in _cords ]

    # 7.1 竖向分组
    # liney [xmin-2, ymin, xmin-2, ymax]
    lines_y = [[c_d[0]+1, c_d[1], c_d[0]+1, c_d[3]]  for c_d in _cords]
    lines_y.sort(key=lambda pt: pt[1])

    ulines_y = liney_union(lines_y, minDot)
    for i, uline_y in enumerate(ulines_y):
        # y方向与uline_y相交的ubox归为一组
        ygrp_uboxes[i] = [ _ubox for _ubox in _uboxes if line_inter_box(uline_y, _ubox, minDot, axis='y') ]
        # 竖向组内，横向排序 （x轴方向从大到小）
        ygrp_uboxes[i].sort(key=lambda pt: pt[2], reverse=True)

    return ygrp_uboxes, ulines_y

def isinter2plus(cord, cords_db_orig_, minDot):
    intercount = 0
    cord1 = xmin,ymin,xmax,ymax = cord[0], cord[1], cord[2], cord[3]  # craft左上,右下
    for cord_ in cords_db_orig_:
        cord2 = xmin_,ymin_,xmax_,ymax_ = cord_[0], cord_[1], cord_[2], cord_[3]  # db左上, 右下
        if rec_interact(cord1, cord2, minDot)[0]:
            intercount += 1
    return intercount>=2

def rng_interact(rng1, rng2):
    interacted = False
    xmin1, xmax1 = rng1[0], rng1[1]
    xmin2, xmax2 = rng2[0], rng2[1]
    if xmin1 <= xmax2 and xmin2 <= xmax1: interacted = True
    rng_u = [ min(xmin1, xmin2), max(xmax1, xmax2)]
    return interacted, rng_u


def get_w_rngs(widths, R=0.04):
    w_sorted = sorted(widths)
    w_rngs_tmp = [ [w*(1-R), w*(1+R)] for w in w_sorted ]
    w_rngs, w_rng = [], w_rngs_tmp.pop(0)
    for _rng in w_rngs_tmp:
        interacted, rng_u = rng_interact(w_rng, _rng)
        if interacted:
            w_rng = rng_u
        else:
            w_rngs.append(w_rng)
            w_rng = _rng
    w_rngs.append(w_rng)
    return w_rngs

def get_line_size(w_rngs, w):
    sizes = ['S','M','L','XL','XXL','XXXL']
    l_sizes = len(sizes)
    for i, rng in enumerate(w_rngs):
        rngl, rngr = rng[0], rng[1]
        if rngl <= w <= rngr:
            if i >= l_sizes: return sizes[-1]
            return sizes[i]
    return sizes[-1]

def re_mapping_lsize(res4api_detect_line):
    # sizes = ['S','M','L','XL','XXL','XXXL']
    sizes = ['XXXL','XXL','XL','L','M','S']
    boxgrp_w = {s:{'avg_w':0,'widths':[],'cnt':0} for s in sizes}
    for i, r_line in enumerate(res4api_detect_line):
        box_line = r_line['box']
        x1,y1, x2,y2, x3,y3, x4,y4 = [ int(cord) for cord in box_line] 
        min_x, max_x = round((x1+x4)/2), round((x2+x3)/2)
        width_line = abs(max_x - min_x)

        line_size = r_line['size']
        boxgrp_w[line_size]['widths'].append(width_line)
        boxgrp_w[line_size]['cnt'] += 1
    
    # 各size数量， 各数量对应size及属性
    cnts_size, cnt_grp = [],{}
    for size, boxgrp in boxgrp_w.items():
        _cnt = boxgrp['cnt']
        if _cnt == 0: continue
        cnts_size.append(_cnt)
        boxgrp['avg_w'] = np.mean(boxgrp['widths'])
        cnt_grp[_cnt] = boxgrp
        cnt_grp[_cnt]['size'] = size
    cnts_size.sort(reverse=True)
    
    # 求S_size和L_size, 后面进行替换为'S', 'M'('L')
    if len(cnts_size) == 1:
        # 所有 r_line['size']设置为S
        for i, r_line in enumerate(res4api_detect_line):
            r_line['size'] = 'M'
        return
    c1,c2 = cnts_size[0], cnts_size[1]
    top1,top2 = cnt_grp[c1], cnt_grp[c2]
    top1_size, top2_size = top1['size'], top2['size']
    top1_w, top2_w = top1['avg_w'], top2['avg_w']

    # S_size, S_w; L_size, L_w
    S_size, S_w = (top1_size, top1_w) if top1_w < top2_w else (top2_size, top2_w)
    top2s = [(top1_size,top1_w), (top2_size,top2_w)]
    top2s.remove((S_size,S_w))
    L_size, L_w = top2s[0]

    # 重选 L_size, L_w
    for s in sizes:
        _cnt = boxgrp_w[s]['cnt']
        if _cnt>0:
            L_size, L_w = s, boxgrp_w[s]['avg_w']
            break
    S_grp, L_grp = [S_size], [L_size]
    
    for size, boxgrp in boxgrp_w.items():
        if boxgrp['cnt']==0: continue
        if S_size==size or L_size==size: continue
        avg_w = boxgrp['avg_w']
        if avg_w <= S_w:
            S_grp.append(size)
        elif avg_w >= L_w:
            L_grp.append(size)
        else:
            if avg_w/S_w < 1.5:
                S_grp.append(size)
            else:
                L_grp.append(size)
    
    # S_grp --> S, L_grp --> L
    for i, r_line in enumerate(res4api_detect_line):
        lsize = r_line['size']
        # jiapi = r_line['jiapi']
        lsize = 'S' if lsize in S_grp else 'M'
        # lsize = 'S' if 'jiapi'==jiapi else 'M'
        r_line['size'] = lsize

def concat_boxes(res4api_detect_line, res4api_detect_line_db, pth_img='', dbg=False):
    filename, pth_sav_dir = '',''
    if not ''==pth_img:
        filename, file_ext = os.path.splitext(os.path.basename(pth_img))
        pth_dir = os.path.abspath( os.path.dirname(pth_img) )
        pth_sav_dir = os.path.join(pth_dir,'output' )
    
    # 〇.分别获取craft和db的api结果
    # 画craft框（红色）  BEGIN
    res_detect_line = {
        int(itm['name']):{'box':[float(pt) for pt in itm['box']], 'text': itm['text'] } for itm in res4api_detect_line
    }
    out_sav = ''
    cords_craft_orig = [v['box'] for index,v in res_detect_line.items()]

    pth_img_rect = os.path.join(pth_sav_dir,filename+'rec.jpg') if not ''==pth_img else ''
   
    # 在craft画框基础上，再画db框（蓝色）  BEGIN
    res_detect_line_db = {
        int(itm['name']):{'box':[float(pt) for pt in itm['box']], 'text': itm['text'] } for itm in res4api_detect_line_db
    }
    cords_db_orig = [v['box'] for index,v in res_detect_line_db.items()]

    # I. 将坐标加上宽,长,宽长比等信息
    cords_craft_orig_ = [ conv_cords(cord) for cord in cords_craft_orig  ]
    cords_db_orig_ = [ conv_cords(cord) for cord in cords_db_orig  ]
    cords_orig_ = cords_craft_orig_ + cords_db_orig_

    # 过滤 宽w/长h > 1.1的框， RATIO_WH_FILTER 参数配置
    RATIO_WH_FILTER = 1.1  # 过滤 w/h > 1.05 ?  的框
    # RATIO_WH_FILTER = 1.5  # 过滤 w/h > 1.05 ?  的框 原宽长比1.1,拟改成1.5

    W_AVG = np.mean([cord[-3] for cord in cords_orig_])
    H_AVG = np.mean([cord[-2] for cord in cords_orig_])
    # 改变初始过滤策略:过滤交叉>=3的框 -- BEGIN
    # cords_orig_ = [ cord for cord in cords_orig_ if cord[-1]<=RATIO_WH_FILTER and cord[-2]>W_AVG]
    cords_orig_ = [ cord for cord in cords_orig_ if cord[-1]<=RATIO_WH_FILTER]

    # if dbg and not pth_img=='':
    if not pth_img=='':
         # 画craft框（红色）  END
        draw_box(cords_craft_orig, pth_img, pth_img_rect )
        # 在craft画框基础上，再画db框（蓝色）  END
        draw_box(cords_db_orig, pth_img_rect, pth_img_rect, color=(255,0,0))

    # 改变初始过滤策略:过滤交叉>=3的框 -- BEGIN
    _Xmin = min([c_d[0] for c_d in cords_orig_])
    _Ymin = min([c_d[1] for c_d in cords_orig_])
    _minDot = (_Xmin, _Ymin)  # 最左上的点

    cords_inter2plus = [cord for cord in cords_craft_orig_ if isinter2plus(cord, cords_db_orig_, _minDot) ]
    for cord in cords_inter2plus:
        try:
            cords_orig_.remove(cord)
        except Exception as e:
            continue
    # 改变初始过滤策略:过滤交叉>=3的框 -- END


    # II. 坐标转换(宽度缩放到0.7)
    cords, cords_craft, cords_db = [], [], []
    # 宽度缩放比例， RESIZE_X 参数配置
    RESIZE_X, RESIZE_Y = 0.7, 0.9

    for cord in cords_orig_:  # x方向坐标压缩（避免横向不同大框之间的框交错）
        _cord = xmin,ymin,xmax,ymax = cord[0], cord[1], cord[2], cord[3]
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

    for cord_db in cords_db_orig:
        _cord_db = int(cord_db[0]), int(cord_db[1]), int(cord_db[4]), int(cord_db[5])
        if RESIZE_X < 1.0:
            _cord_db = list(scale_x(_cord_db, RESIZE_X))
            _cord_db = list(scale_y(_cord_db, RESIZE_Y))
        cords_db.append(_cord_db)



    # 求全局XMIN, YMIN, XMAX, YMAX
    Xmin = min([c_d[0] for c_d in cords])
    Ymin = min([c_d[1] for c_d in cords])
    Xmax = max([c_d[2] for c_d in cords])
    Ymax = max([c_d[3] for c_d in cords])

    # line [xmin, Ymin-2, xmax, Ymin-2]
    lines = [[c_d[0], Ymin-10, c_d[2], Ymin-10]  for c_d in cords]
    # 计算线段平均长度
    line_lens = [abs(l[2]-l[0]) for l in lines]
    LLEN_AVG = np.mean(line_lens)
    RATIO_LINE_FILTER = 2/3
    # 过滤掉过短的线段
    lines = [l for l in lines if (LLEN_AVG*RATIO_LINE_FILTER)<(l[2]-l[0])]
    lines.sort(key=lambda pt: pt[2], reverse=True)

    # if dbg and not pth_img=='':
    if not pth_img=='':
         # 画craft框（红色）  END
        _cords_craft = [[c[0],c[1], c[2],c[1], c[2],c[3], c[0],c[3]] for c in cords_craft]
        _cords_db = [[c[0],c[1], c[2],c[1], c[2],c[3], c[0],c[3]] for c in cords_db ]
        pth_img_rect_resize = os.path.join(pth_sav_dir,filename+'rec_resize.jpg') if not ''==pth_img else ''
        draw_box(_cords_craft, pth_img, pth_img_rect_resize )
        # 在craft画框基础上，再画db框（蓝色）  END
        draw_box(_cords_db, pth_img_rect_resize, pth_img_rect_resize, color=(255,0,0))

    # III. x轴方向线段归并
    lines_union, line_u = [], lines[0]

    for i in range(1,len(lines)):
        _line = lines[i]
        interacted, _line_u, _ = line_interact(line_u, _line, minDot=(Xmin,Ymin))
        if interacted:
            line_u = _line_u
        else:
            lines_union.append(line_u)
            line_u = _line
    lines_union.append(line_u)

    # 画线段
    if dbg:
        pth_img_with_unionline = pth_img_rect.replace('rec.jpg','rec_uline.jpg')
        draw_line(lines_union, pth_img_rect, pth_img_with_unionline, color=(0,255,0), thickness=3)

    # IV. 对于归并后线段，求间隙中竖线位置，从而求得大框（大行）坐标
    # 并将归并后线段、中竖线绘制在图片上
    bigboxes = []
    bigboxes_points = [ [(lines_union[i-1][0] + lines_union[i][2])/2 ,Ymin-5] for i in range(1, len(lines_union)) ]
    # 最右边一个点 + [...] + 最左边一个点
    bigboxes_points = [[lines_union[0][2]+16, Ymin-5]] + bigboxes_points + [[lines_union[-1][0]-16, Ymin-5]]
    for i in range(1, len(bigboxes_points)):
        _xmin, _ymin = bbp = bigboxes_points[i]
        _xmax, _ymin = bbp_pre = bigboxes_points[i-1]
        bigbox = [ _xmin+4, _ymin, _xmax-4, _ymin, _xmax-4, Ymax+5, _xmin+4, Ymax+5]
        bigboxes.append(bigbox)
    # 画大框
    if dbg:
        pth_img_with_bigbox = pth_img_rect.replace('rec.jpg','rec_bigbox.jpg')
        draw_box(bigboxes, pth_img_rect_resize, pth_img_with_bigbox, color=(0,128,0), thickness=2)

    # V. 过滤属于大框内的subbox
    cords.sort(key=lambda pt:pt[0], reverse=True)
    # print(cords)
    bigboxes_subboxes = {
        i: {
            'cords_big': bigbox,  # 四点坐标
            'sub_boxes': filter_subbox(cords, bigbox, minDot=(Xmin,Ymin))  # 左上右下两点坐标
        }
        for i, bigbox in enumerate(bigboxes)
    }
    # 生成随机颜色数组
    colors_box = randcolors( len(bigboxes_subboxes) )
    if dbg:
        # BEGIN - 重新画大框 和 大框包含的子框 （大框和子框颜色相同，大框编号）
        # print(bigboxes_subboxes)
        # 画大框、标数字
        pth_img_subbox = pth_img_rect.replace('rec.jpg','rec_subbox.jpg')
        pth_img_ubox = pth_img_rect.replace('rec.jpg','rec_ubox.jpg')
        
        draw_box([ bigboxes_subboxes[0]['cords_big'] ], pth_img, pth_img_subbox, color=colors_box[0], thickness=1, text='0')
        for i, bigitem in bigboxes_subboxes.items():
            bigbox = bigitem['cords_big']
            _sub_boxes = bigitem['sub_boxes']
            # drawbox
            sub_boxes = [ [b[0],b[1], b[2],b[1],  b[2],b[3], b[0],b[3]] for b in _sub_boxes]
            color = colors_box[i]
            if i>0:
                draw_box([bigbox], pth_img_subbox, pth_img_subbox, color=color, thickness=1, text=str(i))
            draw_box(sub_boxes, pth_img_subbox, pth_img_subbox, color=color, thickness=1)
        # END - 重新画大框 和 大框包含的子框 （大框和子框颜色相同，大框编号）

    # VI. 各大框内的subbox融合归并
    for i, bigitem in bigboxes_subboxes.items():
        sub_boxes = bigitem['sub_boxes']
        uboxes = union_subboxes(sub_boxes, cords_db, minDot=(Xmin,Ymin))
        bigboxes_subboxes[i]['uboxes'] = uboxes

    color_ubox = (0,128,0)
    if dbg:
        # BEGIN - 重新画大框 和 大框包含的融合后子框 （大框和子框颜色相同，大框编号）
        # 画大框、标数字
        pth_img_ubox = pth_img_rect.replace('rec.jpg','rec_ubox.jpg')

        
        # draw_box([ bigboxes_subboxes[0]['cords_big'] ], pth_img, pth_img_ubox, color=colors_box[0], thickness=1, text='0')
        draw_box([ bigboxes_subboxes[0]['cords_big'] ], pth_img, pth_img_ubox, color=color_ubox, thickness=1, text='0')
        for i, bigitem in bigboxes_subboxes.items():
            bigbox = bigitem['cords_big']
            _uboxes = bigitem['uboxes']
            # drawbox
            uboxes = [ [b[0],b[1], b[2],b[1],  b[2],b[3], b[0],b[3]] for b in _uboxes]
            color = color_ubox # colors_box[i]
            if i>0:
                draw_box([bigbox], pth_img_ubox, pth_img_ubox, color=color, thickness=1, text=str(i))
            draw_box(uboxes, pth_img_ubox, pth_img_ubox, color=color, thickness=1)
        # END - 重新画大框 和 大框包含的融合后子框 （大框和子框颜色相同，大框编号）

    
    for i, bigitem in bigboxes_subboxes.items():
        bigbox = bigitem['cords_big']
        _uboxes = bigitem['uboxes']
        # 7.1 竖向分组, 竖向组内，横向排序
        ygrp_uboxes, ulines_y = ygroup_uboxes(_uboxes, minDot=(Xmin,Ymin))
        bigboxes_subboxes[i]['ygrp_uboxes'] = ygrp_uboxes
        bigboxes_subboxes[i]['ulines_y'] = ulines_y
    
    # 绘制每个bigbox框下，y轴方向 ulines_y
    # pth_img_ubox
    ulinesy_all = [ uliney for i, bigitem in bigboxes_subboxes.items() for uliney in bigitem['ulines_y']]

    if dbg:
        pth_img_uliney = pth_img_rect.replace('rec.jpg','rec_ulinesy.jpg')
        draw_line(ulinesy_all, pth_img_ubox, pth_img_uliney, color=(0,255,0), thickness=3)

    # VII. uboxes 整体编号
    idx_ubox_g= 0
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
                _ubox = scale_x(_ubox, resize_x=(10/6))
                # y方向缩放10/9
                _ubox = scale_y(_ubox, resize_y=(10/9))
                bigboxes_subboxes[i]['uboxes_g'][idx_ubox_g] = _ubox
                idx_ubox_g+=1

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
        ub[0],ub[1], ub[2],ub[1], ub[2],ub[3], ub[0],ub[3]
    # ] for (ub,jiapi) in uboxes_g]
    ] for ub in uboxes_g]

    # 把bigbox用暗红色画出来，画在uboxes_g上（对于没有双行夹批的，大框替代所有小框并在x,y方向缩放0.96）
    bigboxes = [bigitem['cords_big'] for i, bigitem in bigboxes_subboxes.items()]

    if not ''==pth_img:
        pth_img_uboxes_g = pth_img_rect.replace('rec.jpg','rec_uboxes_g.jpg')
        draw_box(uboxes_lurd, pth_img, pth_img_uboxes_g, color=color_ubox, thickness=2, seqnum=True)
        draw_box(bigboxes, pth_img_uboxes_g, pth_img_uboxes_g, color=(128,0,0), thickness=1, seqnum=True)

    # VIII. 求框中位数，计算字体M, S（双行夹批）

    return res4api_detect_line_union
    
    # db和craft相交,保留db(例外: craft全包db且远长于db)
    # 重叠矩形(可能是ubox) 判断逻辑