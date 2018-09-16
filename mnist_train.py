# coding:utf-8
import mxnet as mx
import mxnet.ndarray as nd
from mobile_net_v2 import *
from mxnet import gluon as g


def Transform(data, label):
    data = nd.array(data)
    data = mx.image.imresize(data, 112, 112)
    return nd.transpose(data.astype('float'), (2, 0, 1)) / 255.0, nd.array([label]).asscalar().astype('float32')


# #train_root = './dataset_mini\\'
# BATCH_SIZE = 4
# dataset = mx.gluon.data.vision.datasets.ImageFolderDataset(train_root, flag=1,
#                                                            transform=Transform)  # jupyter中如果指定进程数，必须把 加载dataloder 写在 __main__里
# dataloder = mx.gluon.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, last_batch='rollover')

#######################mnist data read#############################
mx.random.seed(42)
mnist = mx.test_utils.get_mnist()
BATCH_SIZE = 100
train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], BATCH_SIZE, shuffle=True)
val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], BATCH_SIZE)

learning_rate = 1e-3
num_epochs = 10
CTX = [mx.gpu(0)]

net = MobilenetV2(10, 1, prefix="")
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=CTX)  # 初始化网络

optimizer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': 1e-5})
metric = mx.metric.Accuracy()
softmax_cross_entropy_loss = g.loss.SoftmaxCrossEntropyLoss()

# for epoch in range(num_epochs):
#     train_data.reset()
#     for batch in train_data:
#         with mx.autograd.record():
#             data = gluon.utils.split_and_load(batch.data[0], ctx_list=CTX, batch_axis=0)
#             label = gluon.utils.split_and_load(batch.label[0], ctx_list=CTX, batch_axis=0)
#             fc_out = net(data.astype('float32'))
#             # loss = nd.softmax_cross_entropy(fc1,label.astype('float32'))
#             loss = softmax_cross_entropy_loss(fc_out, label.astype('float32'))
#             # print loss
#             metric.update(label, fc_out)
#
#         loss.backward()
#         optimizer.step(BATCH_SIZE)
#     name, acc = metric.get()
#     metric.reset()
#     print('training acc at epoch %d: %s=%f' % (epoch + 1, name, acc))


epoch = 10
# Use Accuracy as the evaluation metric.
metric = mx.metric.Accuracy()
softmax_cross_entropy_loss = g.loss.SoftmaxCrossEntropyLoss()
for i in range(epoch):
    # Reset the train data iterator.
    train_data.reset()
    # Loop over the train data iterator.
    for batch in train_data:
        # Splits train data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = g.utils.split_and_load(batch.data[0], ctx_list=CTX, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = g.utils.split_and_load(batch.label[0], ctx_list=CTX, batch_axis=0)
        outputs = []
        # Inside training scope
        with mx.autograd.record():
            for x, y in zip(data, label):
                z = net(x)
                # Computes softmax cross entropy loss.
                loss = softmax_cross_entropy_loss(z, y)
                # Backpropagate the error for one iteration.
                loss.backward()
                outputs.append(z)
        # Updates internal evaluation
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        optimizer.step(batch.data[0].shape[0])
    # Gets the evaluation result.
    name, acc = metric.get()
    # Reset evaluation result to initial state.
    metric.reset()
    print('training acc at epoch %d: %s=%f' % (i, name, acc))
