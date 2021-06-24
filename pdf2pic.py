# -*- coding: utf-8 -*-
# @Time   : 2021/5/28 22:54
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : pdf2pic.py

import PyPDF4
import fitz
import pikepdf
from PIL import Image


# 对pdf文件进行简单的解密
def jiemi(pdfpath):
    new_pdfpath = pdfpath[:-4] + '_new' + pdfpath[-4:]

    fp = open(pdfpath, "rb+")
    pdfFile = PyPDF4.pdf.PdfFileReader(fp)

    # pdf 解密
    if pdfFile.isEncrypted:
        pdf = pikepdf.open(pdfpath, password='')
        pdf.save(new_pdfpath)
    return new_pdfpath

    # 将每一页转化为图片并保存


def pdf_image(pdf_name, Gray=False):
    img_paths = []
    pdf = fitz.Document(pdf_name)
    for i, pg in enumerate(range(0, pdf.pageCount)):
        page = pdf[pg]  # 获得每一页的对象
        trans = fitz.Matrix(3.0, 3.0).preRotate(0)
        pm = page.getPixmap(matrix=trans, alpha=False)  # 获得每一页的流对象
        # pm.writePNG(dir_name + os.sep + base_name[:-4] + '_' + '{:0>3d}.png'.format(pg + 1))  # 保存图片
        img_path = pdf_name[:-4] + '_' + str(pg + 1) + '.jpg'
        pm.writePNG(img_path)  # 保存图片
        img_paths.append(img_path)

        if Gray:  # 是否转为灰度
            img = Image.open(img_path)
            # img.show()
            low = img.convert('L')
            low.save(img_path)

            '''
            这种模式转换的灰度图片size比较大
            img = cv2.imread(img_path, 0)
            # cv2.imshow("img", img)
            cv2.imwrite(img_path, img)
            '''

    pdf.close()
    return img_paths


pdf_image('/Users/Beyoung/Desktop/Projects/AC_OCR/宝庆/宝庆.pdf', Gray=True)
