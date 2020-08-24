import os
import numpy as np
import cv2
import random
from tqdm import tqdm

if __name__ == '__main__':
    random.seed(0)

    data_root_folder = '/home/lhw/czcv_2t_workspace/yz/face_mask'
    img_lst_file = os.path.join(data_root_folder, 'img.txt')   # 所有图片数据（绝对路径）
    train_lst_file = os.path.join(data_root_folder, 'train.txt')
    val_lst_file = os.path.join(data_root_folder, 'val.txt')

    print(train_lst_file)
    print(val_lst_file)
    train_fd = open(train_lst_file, 'w')
    val_fd = open(val_lst_file, 'w')


    with open(img_lst_file, 'r') as f:
        img_lst = f.readlines()
    img_lst = [x for x in img_lst]
    # random.shuffle(img_lst)

    for img_file in tqdm(img_lst):
        if random.random() < 0.7:
            train_fd.write('{}'.format(img_file))
        else:
            val_fd.write('{}'.format(img_file))

    train_fd.close()
    val_fd.close()

    print('finish')
