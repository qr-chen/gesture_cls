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
label_name_dict = {0: 'bad', 1: 'finger-heart', 2: 'first', 3: 'five', 4: 'four', 5: 'good', 6: 'love', 7: 'ok',
                   8: 'phone', 9: 'seven', 10: 'up-gun', 11: 'up-point', 12: 'yeah'}

for image in os.listdir(IMG_ROOT):

    plt.clf()
    img = mx.image.imread(IMG_ROOT + image)
    plt.imshow(img.asnumpy())
    img = Transform_data(img)  # 修改img 通道
    img = nd.expand_dims(img, axis=0)
    img = img.as_in_context(CTX)
    time_start = time.time()
    output = model(img)
    time_end = time.time()
    output = mx.nd.squeeze(output)
    score = mx.nd.softmax(output)
    index = (mx.nd.argmax(score, axis=0)).asnumpy()
    # time_end = time.time()
    label_name = label_name_dict[int(index)]
    label_score = (score[index]).asnumpy()

    plt.title('class predict:%s score:%f' % (label_name, label_score), fontsize=12, color='r')
    plt.savefig(SAVE_ROOT + image)

    print str(time_end - time_start) + 's'
    print 'successfully saved '
print 'done'
