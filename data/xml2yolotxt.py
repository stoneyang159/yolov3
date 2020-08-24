import os
from lxml import etree as ET
from lxml.etree import Element, SubElement, tostring, ElementTree
from tqdm import tqdm
import pdb

'''
@param[in]: size. 图片尺寸 [w, h]
@param[in]: box. roi区域 [x0,y0,x1,y1]
@param[out]: 归一化rect info. [normal_center_x, normal_center_y, normal_w, normal_h]
'''
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[0] + box[1]) / 2.0
    cy = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    ncx = cx * dw
    w = w * dw
    ncy = cy * dh
    h = h * dh
    return (ncx, ncy, w, h)


def convert_annotation(image_add, in_path, out_path, classes):
    image_name = os.path.basename(image_add)
    image_name = image_name.replace('.jpg', '')
    in_file = open(os.path.join(in_path, image_name + '.xml'), 'r')
    out_file = open(os.path.join(out_path, image_name+'.txt'), 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find('size')

    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # 在一个XML中每个Object的迭代
    for obj in root.iter('object'):
        # iter()方法可以递归遍历元素/树的所有子元素
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        # cls_set.add(cls)

        # 如果训练标签中的品种不在程序预定品种，或者difficult = 1，跳过此object
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)  # 这里取索引，避免类别名是中文，之后运行yolo时要在cfg将索引与具体类别配对
        xmlbox = obj.find('bndbox')

        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    classes = ['face', 'face_mask']

    data_root_folder = '/home/lhw/czcv_2t_workspace/yz/face_mask'
    img_folder = os.path.join(data_root_folder, 'JPEGImages')
    xml_folder = os.path.join(data_root_folder, 'Annotations')
    yolotxt_folder = os.path.join(data_root_folder, 'labels')

    if not os.path.exists(yolotxt_folder):
        os.makedirs(yolotxt_folder)

    img_lst_file = os.path.join(data_root_folder, 'img.txt')   # 所有图片数据（绝对路径）
    if not os.path.exists(img_lst_file):
        img_lst = [os.path.join(img_folder, x) for x in os.listdir(img_folder) if 'jpg' in x]
        with open(img_lst_file, 'w') as f:
            for img_file in img_lst:
                f.write("{}\n".format(img_file))

    else:
        with open(img_lst_file, 'r') as f:
            img_lst = f.readlines()
            img_lst = [x.split('\n')[0] for x in img_lst]

    print('converting...')
    for image_add in tqdm(img_lst):
        image_add = image_add.strip()
        convert_annotation(image_add, xml_folder, yolotxt_folder, classes)

    print("Finished")

