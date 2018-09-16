# -*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
from mobile_net_v2 import *
from model_utils import *
import time

IMG_ROOT = './test_data/'
SAVE_ROOT = './test_results/'
CTX = mx.gpu(0)
model = MobilenetV2(13, 0.25)
model.hybridize()
model.load_params('./model_saved/gesture13_final.params', ctx=CTX)
label_name_dict = {0: 'bad', 1: 'finger-heart', 2: 'fist', 3: 'five', 4: 'four', 5: 'good', 6: 'love', 7: 'ok',
                   8: 'phone', 9: 'seven', 10: 'up-gun', 11: 'up-point', 12: 'yeah'}

for image in os.listdir(IMG_ROOT):
    img = mx.image.imread(IMG_ROOT + image)
    img = Transform_data(img)  # 修改img 通道
    img = nd.expand_dims(img, axis=0)
    img = img.as_in_context(CTX)
    output = model(img)
    model.export("./model_saved/gesture13_0.25_112")



    print 'successfully saved '
    break
print 'done'
