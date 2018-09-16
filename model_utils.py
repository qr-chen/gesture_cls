# -*- coding:utf-8 -*-
import mxnet as mx
import mxnet.ndarray as nd
mx.random.seed(42)  # set seed for repeatability

def Transform_data(data):
    data = nd.array(data)
    data = mx.image.imresize(data, 112, 112)  # 原始采用 112*112
    return nd.transpose(data.astype('float32'), (2, 0, 1)) / 255.0


def Transform_data_label(data, label):
    data = nd.array(data)
    data = mx.image.imresize(data, 112, 112)
    return nd.transpose(data.astype('float32'), (2, 0, 1)) / 255.0, nd.array([label]).asscalar().astype('float32')


def AUG_Transform_data_label(data, label):
    data = nd.array(data)
    data = mx.image.imresize(data, 112, 112)
    light_aug = mx.gluon.data.vision.transforms.RandomLighting(0.3)
    bright_aug = mx.gluon.data.vision.transforms.RandomBrightness(0.3)
    flip_aug = mx.gluon.data.vision.transforms.RandomFlipLeftRight()
    data = bright_aug(light_aug(flip_aug(data)))
    return nd.transpose(data.astype('float32'), (2, 0, 1)) / 255.0, nd.array([label]).asscalar().astype('float32')
