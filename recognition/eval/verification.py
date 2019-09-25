"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd

np.set_printoptions(suppress=True, threshold=1000000)

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def calculate_cos(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    """

    :param thresholds: 阈值
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param nrof_folds: K折检测的，分K份检测
    :param pca:
    :return:
    """

    # 数据组合的验证
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    # 获得最小长度标签个数和向量个数中
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])

    # 获得阈值个数
    nrof_thresholds = len(thresholds)

    #
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)
    print('k_fold  ',k_fold)
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    # 如果没有启动pca
    if pca == 0:
        print(embeddings1.shape)
        print(embeddings2.shape)

        num = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        denom = np.linalg.norm(embeddings1,axis=1) * np.linalg.norm(embeddings2,axis=1)
        print('num',num.shape)
        print('denom',denom.shape)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        #print(sim)
    # 分k则进行评估
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #print('train_set', train_set)
        #print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold,1-sim[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 1-sim[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], 1-sim[test_set],
                                                      actual_issame[test_set])
    print('threshold', 1-thresholds[best_threshold_index])
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy



def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    """

    :param thresholds: 阈值
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param nrof_folds: K折检测的，分K份检测
    :param pca:
    :return:
    """
    print('='*50)
    # 数据组合的验证
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    # 获得最小长度标签个数和向量个数中
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])

    # 获得阈值个数
    nrof_thresholds = len(thresholds)

    #
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)
    print('k_fold  ',k_fold)
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    # 如果没有启动pca
    if pca == 0:
        # 求范数距离欧式距离
        diff = np.subtract(embeddings1, embeddings2)# 做减法
        dist = np.sum(np.square(diff), 1)# 求平方和

    # 分k则进行评估
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #print('train_set', train_set)
        #print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        print('best_threshold: ', thresholds[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #print(fold_idx)
        #print(train_set)
        #print(test_set)
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        #$print(thresholds)

        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    #print(actual_issame)
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    #print(true_accept, false_accept)
    #print(n_same, n_diff)

    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """

    :param embeddings:
    :param actual_issame:
    :param nrof_folds: K折检测的，分K份检测
    :param pca:
    :return:
    """
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)

    # 取所有的第一张图片的向量
    embeddings1 = embeddings[0::2]
    # 取所有的第二张图片的向量
    embeddings2 = embeddings[1::2]


    # tpr:正类预测正确（召回率）
    # fpr:正类预测错误
    # accuracy:准确率
    #tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
    #                                 np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)

    thresholds = np.arange(0, 1, 0.001)
    tpr, fpr, accuracy = calculate_cos(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, pca=0)

    thresholds = np.arange(0, 4, 0.001)
    #print(embeddings1.shape)
    #print(embeddings2.shape)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


# 加载bin数据
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = nd.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return (data_list, issame_list)


def test(data_set, mx_model, batch_size, nfolds=10, data_extra=None, label_shape=None):
    """

    :param data_set: 测试数据
    :param mx_model: 测试模型
    :param batch_size: 测试批次大小
    :param nfolds: K折检测的，分K份检测
    :param data_extra:
    :param label_shape: 标签数据的形状
    :return:
    """
    print('testing verification..')
    data_list = data_set[0] # 两张图片的像素
    issame_list = data_set[1] # 标签
    model = mx_model #模型
    embeddings_list = [] # 输出特征向量

    # 如果存在额外数据
    if data_extra is not None:
        _data_extra = nd.array(data_extra)

    # 记录测试消耗时间
    time_consumed = 0.0

    # 如果标签的形状没有设定
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)

    # 对每个测试集进行测试
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0 # 记录训练呢多少批次
        while ba < data.shape[0]:
            #print('data.shape[0]',data.shape[0])
            # 防止超出界限，最后一次取最小的
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba

            # 切割数据，得到一个batch_size的数据
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)

            #print(_data.shape, _label.shape)
            time0 = datetime.datetime.now()

            # 如果有额外测试数据_data_extra，则进行添加
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))

            # 传入数据，进行前向传播
            model.forward(db, is_train=False)

            # 获得输出
            net_out = model.get_outputs()

            # _arg, _aux = model.get_params()
            # __arg = {}
            # for k,v in _arg.iteritems():
            #  __arg[k] = v.as_in_context(_ctx)
            # _arg = __arg
            # _arg["data"] = _data.as_in_context(_ctx)
            # _arg["softmax_label"] = _label.as_in_context(_ctx)
            # for k,v in _arg.iteritems():
            #  print(k,v.context)
            # exe = sym.bind(_ctx, _arg ,args_grad=None, grad_req="null", aux_states=_aux)
            # exe.forward(is_train=False)
            # net_out = exe.outputs

            # 获得输出的向量
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            # print(_embeddings.shape)

            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0

    # 对每个测试数据测试出来的结果进行评估
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            # 求向量的范数：https://blog.csdn.net/jack339083590/article/details/79171585
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    #print(len(embeddings_list[0]))
    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    # _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
    # acc1, std1 = np.mean(accuracy), np.std(accuracy)

    # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    # embeddings = np.concatenate(embeddings_list, axis=1)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)

    # 对输出向量进行评估
    #print(embeddings[0].shape,issame_list)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)

    # 对准确率进行平均化和标准化，标准化表示测试的稳定程度
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def test_badcase(data_set, mx_model, batch_size, name='', data_extra=None, label_shape=None):
    print('testing verification badcase..')
    data_list = data_set[0]
    issame_list = data_set[1]
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            # print(_data.shape, _label.shape)
            time0 = datetime.datetime.now()




            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    thresholds = np.arange(0, 4, 0.01)
    actual_issame = np.asarray(issame_list)
    nrof_folds = 10
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    data = data_list[0]

    pouts = []
    nouts = []

    cout = 0
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #print('train_set',train_set.shape)
        #print('train_set',train_set.shape)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        # print(train_set)
        # print(train_set.__class__)
        for threshold_idx, threshold in enumerate(thresholds):
            p2 = dist[train_set]
            p3 = actual_issame[train_set]
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, p2, p3)
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])
        best_threshold = thresholds[best_threshold_index]
        num = 0
        for iid in test_set:
            num += 1
            ida = iid * 2
            idb = ida + 1
            asame = actual_issame[iid]
            _dist = dist[iid]
            violate = _dist - best_threshold
            if not asame:
                violate *= -1.0
            if violate > 0.0:

                #print(cout * len(test_set) + num)
                imga = data[ida].asnumpy().transpose((1, 2, 0))[..., ::-1]  # to bgr
                imgb = data[idb].asnumpy().transpose((1, 2, 0))[..., ::-1]
                # print(imga.shape, imgb.shape, violate, asame, _dist)
                if asame:
                    pouts.append((imga, imgb, _dist, best_threshold, ida))
                else:
                    nouts.append((imga, imgb, _dist, best_threshold, ida))

        cout += 1
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    acc = np.mean(accuracy)
    pouts = sorted(pouts, key=lambda x: x[2], reverse=True)
    nouts = sorted(nouts, key=lambda x: x[2], reverse=False)
    print(len(pouts), len(nouts))
    print('acc', acc)
    gap = 10
    image_shape = (112, 224, 3)
    out_dir = "./badcases"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if len(nouts) > 0:
        threshold = nouts[0][3]
    else:
        threshold = pouts[-1][3]

    for item in [(pouts, 'positive(false_negative).png'), (nouts, 'negative(false_positive).png')]:
        cols = 4
        rows = 8000
        outs = item[0]
        if len(outs) == 0:
            continue
        # if len(outs)==9:
        #  cols = 3
        #  rows = 3

        _rows = int(math.ceil(len(outs) / cols))
        rows = min(rows, _rows)
        hack = {}

        if name.startswith('cfp') and item[1].startswith('pos'):
            hack = {0: 'manual/238_13.jpg.jpg', 6: 'manual/088_14.jpg.jpg', 10: 'manual/470_14.jpg.jpg',
                    25: 'manual/238_13.jpg.jpg', 28: 'manual/143_11.jpg.jpg'}

        filename = item[1]
        if len(name) > 0:
            filename = name + "_" + filename
        filename = os.path.join(out_dir, filename)
        img = np.zeros((image_shape[0] * rows + 20, image_shape[1] * cols + (cols - 1) * gap, 3), dtype=np.uint8)
        img[:, :, :] = 255
        text_color = (0, 0, 153)
        text_color = (255, 178, 102)
        text_color = (153, 255, 51)
        for outi, out in enumerate(outs):
            row = outi // cols
            col = outi % cols
            if row == rows:
                break
            imga = out[0].copy()
            imgb = out[1].copy()
            if outi in hack:
                idx = out[4]
                print('noise idx', idx)
                aa = hack[outi]
                imgb = cv2.imread(aa)
                # if aa==1:
                #  imgb = cv2.transpose(imgb)
                #  imgb = cv2.flip(imgb, 1)
                # elif aa==3:
                #  imgb = cv2.transpose(imgb)
                #  imgb = cv2.flip(imgb, 0)
                # else:
                #  for ii in range(2):
                #    imgb = cv2.transpose(imgb)
                #    imgb = cv2.flip(imgb, 1)
            dist = out[2]
            _img = np.concatenate((imga, imgb), axis=1)
            k = "%.3f" % dist
            # print(k)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(_img, k, (80, image_shape[0] // 2 + 7), font, 0.6, text_color, 2)
            # _filename = filename+"_%d.png"%outi
            # cv2.imwrite(_filename, _img)
            img[row * image_shape[0]:(row + 1) * image_shape[0],
            (col * image_shape[1] + gap * col):((col + 1) * image_shape[1] + gap * col), :] = _img
        # threshold = outs[0][3]
        font = cv2.FONT_HERSHEY_SIMPLEX
        k = "threshold: %.3f" % threshold
        cv2.putText(img, k, (img.shape[1] // 2 - 70, img.shape[0] - 5), font, 0.6, text_color, 2)
        cv2.imwrite(filename, img)


def dumpR(data_set, mx_model, batch_size, name='', data_extra=None, label_shape=None):
    print('dump verification embedding..')
    data_list = data_set[0]
    issame_list = data_set[1]
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            # print(_data.shape, _label.shape)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    actual_issame = np.asarray(issame_list)
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir',
                        default='E:/1.PaidOn/1.FaceRecognition/2.Dataset/2.PaidOnData/3.TargetData/1.targe',
                        help='')

    # 可以加载多个model，以|隔开，如'../../models/model-y1-test2/model,0|1|2'
    #parser.add_argument('--model',default='../../models/model-y1-test2/model,0',help='path to load model.')

    parser.add_argument('--model',default='../models/y1-arcface-emore_lrle-5/model,7',help='path to load model.')

    #parser.add_argument('--target', default='CASIA-FaceV5,lfw,cfp_ff,cfp_fp,agedb_30', help='test targets.')
    parser.add_argument('--target', default='shunde', help='test targets.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=48, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--mode', default=0, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    args = parser.parse_args()
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
    import src.common.face_image as face_image

    prop = face_image.load_property(args.data_dir)
    image_size = prop.image_size
    print('image_size', image_size)
    ctx = mx.gpu(args.gpu)
    nets = []

    # ../../models/model-y1-test2/model,0
    vec = args.model.split(',')
    prefix = args.model.split(',')[0]
    epochs = []

    # 模型验证
    # 如果没有指定模型后面的后缀，如0
    if len(vec) == 1:
        pdir = os.path.dirname(prefix)

        # 对指定的模型目录查找
        for fname in os.listdir(pdir):
            if not fname.endswith('.params'):
                continue
            _file = os.path.join(pdir, fname)
            if _file.startswith(prefix):
                epoch = int(fname.split('.')[0].split('-')[1])
                epochs.append(epoch)
        epochs = sorted(epochs, reverse=True)

        if len(args.max) > 0:
            _max = [int(x) for x in args.max.split(',')]
            assert len(_max) == 2
            if len(epochs) > _max[1]:
                epochs = epochs[_max[0]:_max[1]]
    else:
        epochs = [int(x) for x in vec[1].split('|')]
    print('model number', len(epochs))
    time0 = datetime.datetime.now()

    # 加载每个epoch的模型，即可以加载多个
    for epoch in epochs:
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        # arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
        # 提取fc1_output层
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)

        # 对输入进行绑定
        # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])

        # 对参数进行设定
        model.set_params(arg_params, aux_params)
        nets.append(model)
    time_now = datetime.datetime.now()
    diff = time_now - time0

    # 完成所有模型的加载，所有模型都保存在nets。注意这里的所有模型是同一网络的模型，
    # 只是后缀epoch不相同而已
    print('model loading time', diff.total_seconds())

    # 保存加载.bin,其中保存则data_set，data_set有两个元素，
    # 一个表示两张图片像素，一个表示两张图片是否相同
    ver_list = []

    ver_name_list = [] # 保存.bin文件的前缀
    for name in args.target.split(','):
        path = os.path.join(args.data_dir, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)

    #  如果args.mode ==0，默认设置为0
    if args.mode == 0:
        # 对每一个训练集数据进行测试
        for i in range(len(ver_list)):
            results = []
            # 对每一个模型进行测试
            for model in nets:
                # ver_list[i]，训练集测试数据列表，args.nfolds参数大小
                acc1, std1, acc2, std2, xnorm, embeddings_list = test(ver_list[i], model, args.batch_size, args.nfolds)
                print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
                print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
                print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
                results.append(acc2)
            print('Max of [%s] is %1.5f' % (ver_name_list[i], np.max(results)))
    elif args.mode == 1:
        model = nets[0]
        test_badcase(ver_list[0], model, args.batch_size, args.target)
    else:
        model = nets[0]
        dumpR(ver_list[0], model, args.batch_size, args.target)


