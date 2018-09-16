# -*- coding:utf-8 -*-
from mxnet import gluon as g
from mobile_net_v2 import *
from mxnet.gluon import utils as gutils
from model_utils import *
from multiprocessing import cpu_count
import numpy as np
import json

TRAIN_ROOT = './train_dataset/'
VAL_ROOT = './val_dataset/'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 600
CTX = [mx.gpu(0), mx.gpu(1)]
epoch_loss = {}
epoch_train_acc = {}
epoch_val_acc = {}

train_dataset = g.data.vision.datasets.ImageFolderDataset(TRAIN_ROOT, flag=1, transform=AUG_Transform_data_label)
train_dataloder = g.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                    num_workers=cpu_count(), shuffle=True, last_batch='rollover')
val_dataset = g.data.vision.datasets.ImageFolderDataset(VAL_ROOT, flag=1, transform=AUG_Transform_data_label)
val_dataloder = g.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  num_workers=cpu_count(), shuffle=True, last_batch='rollover')

model = MobilenetV2(13, 0.125, prefix="")
model.hybridize()
model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=CTX)  # 初始化网络
# net.load_params('./mbnv2-0000.params')
optimizer = mx.gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': LEARNING_RATE, 'wd': 1e-5})
train_metric = mx.metric.Accuracy()
val_metric = mx.metric.Accuracy()
softmax_cross_entropy_loss = g.loss.SoftmaxCrossEntropyLoss()

print train_dataset.synsets
print val_dataset.synsets

for epoch in range(EPOCHS):
    loss_list = []
    for img, label in train_dataloder:
        gpu_imgs = gutils.split_and_load(img, CTX)
        gpu_labels = gutils.split_and_load(label, CTX)
        with mx.autograd.record():

            fc1_lst = [model(img.astype('float32')) for img in gpu_imgs]
            label_lst = [label.astype('float32') for label in gpu_labels]
            # loss = nd.softmax_cross_entropy(fc1,label.astype('float32'))
            loss = [softmax_cross_entropy_loss(fc1, label) for fc1, label in zip(fc1_lst, label_lst)]
            # loss = softmax_cross_entropy_loss(fc1,label)
            # print loss
            train_metric.update(label, fc1)

        for l in loss:
            l.backward()
        optimizer.step(BATCH_SIZE)
        mx.nd.waitall()
        loss_list.append(sum(sum(loss)).asnumpy() / BATCH_SIZE)
    if ((epoch + 1) % 20 == 0 and (epoch + 1) != EPOCHS):
        model.save_params('./model_saved/gesture13_%d.params' % (epoch + 1))
    train_name, train_acc = train_metric.get()
    train_metric.reset()
    epoch_mean_loss = np.mean(loss_list)
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, EPOCHS, epoch_mean_loss))
    epoch_loss['epoch_%d' % (epoch + 1)] = float(epoch_mean_loss)

    for img, label in val_dataloder:
        gpu_imgs = gutils.split_and_load(img, CTX)
        gpu_labels = gutils.split_and_load(label, CTX)
        fc1 = [model(img.astype('float32')) for img in gpu_imgs]
        label = [label.astype('float32') for label in gpu_labels]
        for label, fc1 in zip(label, fc1):
            val_metric.update(label, fc1)

    val_name, val_acc = val_metric.get()
    val_metric.reset()
    print('train acc at epoch %d: %s=%f' % (epoch + 1, train_name, train_acc))
    print('val acc at epoch %d: %s=%f' % (epoch + 1, val_name, val_acc))
    epoch_train_acc['epoch_%d' % (epoch + 1)] = float(train_acc)
    epoch_val_acc['epoch_%d' % (epoch + 1)] = float(val_acc)

model.save_params('./model_saved/gesture13_final.params')
with open('./log/loss_log.json', 'w') as js_f:
    js_f.write(json.dumps(epoch_loss))
with open('./log/train_acc_log.json', 'w') as js_f:
    js_f.write(json.dumps(epoch_train_acc))
with open('./log/val_acc_log.json', 'w') as js_f:
    js_f.write(json.dumps(epoch_val_acc))
