# -*- coding: utf-8 -*-
# @Time   : 2022/1/25 16:52
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : count_pic.py

done_file_txt_list = [
    '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_1.txt',  # 第一批 x-12.1 60w
    '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_2.txt',  # 第二批 12.1-
    '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_3.txt',  # 第三批 12.20-
    # '/disks/sdb/euphoria/deep-text-recognition-benchmark/extract/todo_list_1222_1.txt',  # 第一批 12.22
]
count = 0
for file in done_file_txt_list:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    count += len(lines)
print(count)


