import numpy as np
import random
from PIL import Image
import tensorflow as tf
import cv2


def annotation_list(dir_txt, batch):
    dir_list = []
    my_list = []

    with open(dir_txt) as file:
        for row in file.readlines():
            dir_list.append(row.rstrip())
    for every_folder in dir_list:
        with open('data/tiny-imagenet-200/train/' + every_folder + '/' + every_folder + '_boxes.txt') as f:
            lines = [line.rstrip() for line in f]
        for i in lines:
            my_list.append(i.split())
    random.Random(4).shuffle(my_list)
    return np.asarray(my_list)[batch, 0], np.asarray(my_list)[batch, 1:]


# def images_list(pre_list):
#     np_img_list = []
#     for i in pre_list:
#         np_img = np.asarray(Image.open(('data/tiny-imagenet-200/train/' + i.split('_')[0] + '/images/' + i)),
#                             dtype=np.float64)
#         np_img_list.append(np_img)
#     return np.asarray(np_img_list)


def images_list(pre_list):
    cv_img_list = []
    for i in pre_list:
        cv_img = cv2.imread('data/tiny-imagenet-200/train/' + i.split('_')[0] + '/images/' + i)
        cv_img_list.append(cv_img)
    return np.asarray(cv_img_list)


def set_labels(pre_list):
    labels_list = []
    for item in pre_list:
        labels_list.append(item.split('_')[0])
    return np.asarray(labels_list)


def val_list(val_dir, batch):
    my_list = []
    with open(val_dir) as f:
        lines = [line.rstrip() for line in f]
        for i in lines:
            my_list.append(i.split())
    return np.asarray(my_list)[batch, 0], np.asarray(my_list)[batch, 1], np.asarray(my_list)[batch, 2:]


def np_val_images(pre_list):
    my_list = []
    img_dir = 'data/tiny-imagenet-200/val/images/'
    images_dir = [img_dir + item for item in pre_list]
    for img in images_dir:
        np_img = np.array(Image.open(img))
        my_list.append(np_img)
    return np.asarray(my_list)


def convert2int(pre_list):
    my_list = []
    for i in pre_list:
        my_list.append(tf.strings.to_number(i, tf.float32))
    return np.asarray(my_list)


def label_dic(pre_list):
    list1 = np.arange(0, 200)
    list2 = []
    list3 = []
    int_label = []
    with open('data/tiny-imagenet-200/wnids.txt') as f:
        lines = [line.rstrip() for line in f]
    for i in lines:
        list2.append(i.split('\t')[0])
    my_dic = dict(zip(list2, list1))
    for j in pre_list:
        list3.append(j.split('_')[0])
    for label in list3:
        int_label.append(my_dic[str(label)])
    return np.asarray(int_label)
