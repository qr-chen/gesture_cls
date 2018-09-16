# -*- coding:utf-8 -*-
import numpy as np
import os

DATA_PATH = '/home/users/qiangrui.chen/pycharm_project/hand_cls/train_dataset/yeah/'
TEST_DATA_PATH = '/home/users/qiangrui.chen/pycharm_project/hand_cls/val_dataset/yeah/'

DATA_LST = os.listdir(DATA_PATH)
NUM_IMG = len(DATA_LST)
RATIO = 0.08  # 测试集占比
rand_lst = np.random.choice(NUM_IMG, int(RATIO * NUM_IMG),replace=False)

for index in rand_lst:
    img = DATA_LST[index]
    os.rename(DATA_PATH + img, TEST_DATA_PATH + img)
print '%d done' % int(RATIO * NUM_IMG)
