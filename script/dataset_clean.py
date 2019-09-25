#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import shutil
import copy
import traceback
import numpy as  np
import sklearn
from mxnet import ndarray as nd
from sklearn.cluster import DBSCAN

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

class Model():
    def __init__(self,image_size, args):
        prefix, epoch = args.model.split(',')
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        self.model = mx.mod.Module(symbol=sym, context=mx.gpu())
        self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        self.model.set_params(arg_params, aux_params)

    def predict(self,img):
        img = nd.array(img)
        img = nd.transpose(img, axes=(2, 0, 1)).astype('float32')
        img = nd.expand_dims(img, axis=0)
        #print(img.shape)
        db = mx.io.DataBatch(data=(img,))

        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        embedding = net_out[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding)
        return embedding

def imgs_detect(args,embeddings):
    #print(imgpaths)
    #print(embeddings.shape)
    #print('-'*50)
    if args.detect_mode==1:
        """
       emb_mean = np.mean(embeddings, axis=0, keepdims=True)
      emb_mean = sklearn.preprocessing.normalize(emb_mean)
      sim = np.dot(embeddings, emb_mean.T)
      #print(sim.shape)
      sim = sim.flatten()
      #print(sim.flatten())
      x = np.argsort(sim)
      for ix in range(len(x)):
        _idx = x[ix]
        _sim = sim[_idx]
        #if ix<int(len(x)*0.3) and _sim<args.threshold:
        if _sim<args.threshold:
          continue
        contents.append(ocontents[_idx])
       """
        pass
    else:
        # 最低1.25，建议设置为1以下
        y_pred = DBSCAN(eps = args.threshold, min_samples = 1).fit_predict(embeddings)
        #print(y_pred)
        gmap = {}
        for _idx in range(embeddings.shape[0]):
            label = int(y_pred[_idx])
            if label not in gmap:
                gmap[label] = []
            gmap[label].append(_idx)
        #print(gmap)
        assert len(gmap) > 0
        _max = [0, 0]
        conut = 0
        for label in range(10):
            if not label in gmap:
                break
            glist = gmap[label]
            conut += 1
            if len(glist) > _max[1]:
                _max[0] = label
                _max[1] = len(glist)
        if (_max[1] > 0) and (conut is not 1):
            gmap.pop(_max[0])
            return gmap




def list_image(root, recursive, exts):
    """Traverses the root of directory that contains images and
    generates image list iterator.
    Parameters
    ----------
    root: string
    recursive: bool
    exts: string
    Returns
    -------
    image iterator that contains all the image under the specified path
    """

    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1

def write_list(path_out, image_list):
    """Hepler function to write image list into the file.
    The format is as below,
    integer_image_index \t float_label_index \t path_to_image
    Note that the blank between number and tab is only used for readability.
    Parameters
    ----------
    path_out: string
    image_list: list
    """
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)


def make_list(args):
    """Generates .lst file.
    Parameters
    ----------
    args: object that contains all the arguments
    """
    image_list = list_image(args.indir, args.recursive, args.exts)
    image_list = list(image_list)
    path = os.path.join(args.indir,'.lst')
    write_list(path, image_list)

def read_list(path_in):
    """Reads the .lst file and generates corresponding iterator.
    Parameters
    ----------
    path_in: string
    Returns
    -------
    item iterator that contains information in .lst file
    """
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            # check the data format of .lst file
            if line_len < 3:
                print('lst should have at least has three parts, but only has %s parts for %s' % (line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item

def image_encode(args, i, item, q_out):
    """Reads, preprocesses, packs the image and put it back in output queue.
    Parameters
    ----------
    args: object
    i: int
    item: list
    q_out: queue
    """
    # 获得图片完整的路径
    fullpath = os.path.join(args.indir, item[1])

    # 读取图片像素
    try:
        img = cv2.imread(fullpath, args.color)
    except:
        traceback.print_exc()
        print('imread error trying to load file: %s ' % fullpath)
        q_out.put((i, None, item))
        return
    if img is None:
        print('imread read blank (None) image for file: %s' % fullpath)
        q_out.put((i, None, item))
        return

    # 变换输出图像的大小
    if args.resize:
        if img.shape[0] > img.shape[1]:
            newsize = (args.resize, img.shape[0] * args.resize // img.shape[1])
        else:
            newsize = (img.shape[1] * args.resize // img.shape[0], args.resize)
        img = cv2.resize(img, newsize)

    try:
        # 放入队列
        q_out.put((i, img, item))
    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath, e)
        q_out.put((i, None, item))
        return

def read_worker(args, q_in, q_out):
    """Function that will be spawned to fetch the image
    from the input queue and put it back to output queue.
    Parameters
    ----------
    args: object
    q_in: queue
    q_out: queue
    """
    while True:
        # 获取任务
        deq = q_in.get()
        # 判断该任务是否为None
        if deq is None:
            break

        # i代表的是分配的第几个任务，item包含了第几章图片，路径，所属类别
        i, item = deq

        # 读取图片进行解吗
        image_encode(args, i, item, q_out)


def err_imgs_handle(args, err_msg, f_err, id_imgspath):
    id_imgsdir = id_imgspath[0].replace('\\', '/').split('/')[0]
    print(id_imgsdir)
    print(err_msg)
    f_err.writelines(id_imgsdir + '\n')
    f_err.writelines(str(err_msg) + '\n\n')

    for key in err_msg:
        for _idx in err_msg[key]:
            _id_imgspath = id_imgspath[_idx].strip().replace('\\', '/')
            old_file = os.path.join(args.indir, _id_imgspath)
            new_dir = os.path.join(args.outdir, id_imgsdir)
            new_file = os.path.join(args.outdir, _id_imgspath)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)

            shutil.copyfile(old_file, new_file)
            if args.delete:
                os.remove(old_file)



def write_worker(q_out, args):
    """Function that will be spawned to fetch processed image
    from the output queue and write to the .rec file.
    Parameters
    ----------
    q_out: queue
    """
    print(args.indir)
    f_err = open(os.path.join(args.indir,'like_error.txt'),'w')

    model = Model([112, 112], args)

    buf = {}
    more = True # 队列处理完成，设置为None
    pre_id = 0 # 记录前一张图片的ID，如果前一张，和当前的ID不同，表示前一个人的所有图片已经预测完成
    count = 0

    # 每个id的特征向量
    id_embeddings = None

    # 保存所有图像的路径
    id_imgspath = []

    # 开始产生开始的时间
    pre_time = time.time()

    while more:
        deq = q_out.get()

        # 如果取出信息不为空，则对其中的img进行预测
        if deq is not None:
            # i代表第几个任务 img表示像素
            # item包含了，第几章图片，路径，所属ID
            i, img, item = deq

            # 为了保存所有的任务没有遗漏
            buf[i] = (img, item)
            #print(i)

        # 如果为空，说明已经预测完了所有图片
        else:
            err_msg = imgs_detect(args, id_embeddings)
            if err_msg is not None:
                err_imgs_handle(args, err_msg, f_err, id_imgspath)
            more = False

        while count in buf:
            img, item = buf[count]
            img_path = item[1]
            # print(item)
            del buf[count]
            if img is not None:
                # 对一张图片进行特征向量预测
                embedding = model.predict(img)

                # 判断单个ID的所有图片是否全部预测完成
                if (int(pre_id) != int(item[2])):
                    err_msg = imgs_detect(args, id_embeddings)
                    if err_msg is not None:
                        err_imgs_handle(args, err_msg, f_err, id_imgspath)

                    id_embeddings = None
                    id_imgspath.clear()

                id_imgspath.append(img_path + '\t')
                # 如果id_embeddings为None，说明是每个ID的第一张图片
                if id_embeddings is None:
                    # 第一张图片的特征向量直接赋值即可
                    id_embeddings = embedding
                # 不是第一张，则进行水平拼接
                else:
                    id_embeddings = np.vstack((id_embeddings, embedding))

                # 把当前的ID记录下来
                pre_id = int(item[2])

            # 时间打印
            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1

    f_err.close()




def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')


    parser.add_argument('--indir', default='F:/3.image_classer_new' ,help='path to folder containing images.')

    # 把要删除的图片拷贝到改目录，可以人为进行确认，是否有必要删除
    parser.add_argument('--outdir', default='F:/4.image_clean' ,help='path to folder containing images.')

    # 预训练模型的路径
    parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')

    # 判定是否为同一人的相似度阈值
    parser.add_argument('--threshold', default=0.85, type=float, help='')

    cgroup = parser.add_argument_group('Options for creating image lists')


    # 该处填写为True，则代表是为indir对应的目录下生成.lst文件，注意，此时不会进行人脸分类
    cgroup.add_argument('--list', default=False,help='')

    # 需要分类图片的格式
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png','.bmp'],
                        help='list of acceptable image extensions.')

    # 是否进行递归搜索
    cgroup.add_argument('--recursive', default=True,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')


    # 进行分类相关的参数
    rgroup = parser.add_argument_group('Options for creating database')

    # 检测出来人脸保存的大小
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')

    #设置筛选的人脸的模式，目前只有聚类一种方式
    parser.add_argument('--detect_mode', default = 0, type=int, help='' )

    parser.add_argument('--delete', default=False, type=bool, help='')

    # 读取图片，进行解码的线程数目
    rgroup.add_argument('--num-thread', type=int, default=4,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')

    # 读取的颜色格式，一般选取默认值
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')

    # 需要生成图片的格式
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png','.bmp'],
                        help='specify the encoding of the images.')



    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # if the '--list' is used, it generates .lst file
    if args.list:
        make_list(args)
    # otherwise read .lst file to generates .rec file
    else:
        # 列出输入目录的所有文件
        files = [os.path.join(args.indir, fname) for fname in os.listdir(args.indir)
                    if os.path.isfile(os.path.join(args.indir, fname))]

        count = 0
        for fname in files:
            # 判断该文件是否以.lst结尾
            if fname.endswith('.lst'):
                count += 1
                # 读取文件的所有内容
                image_list = read_list(fname)

                # -- write_record -- #
                # 为每个输入进程创建一个队列
                q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                # 创建单个输出队列
                q_out = multiprocessing.Queue(1024)

                # define the process
                # 定义输入进程，用来读取图片
                read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                                for i in range(args.num_thread)]

                # process images with num_thread process
                # 启动所有输入进程，开始读取图片
                for p in read_process:
                    p.start()

                # only use one process to write .rec to avoid race-condtion
                # 创建一个读取进程。并且启动
                write_process = multiprocessing.Process(target=write_worker, args=(q_out, args))
                write_process.start()


                # put the image list into input queue
                # 循环把要读取的图片，分配给每一个读取进程
                for i, item in enumerate(image_list):
                    q_in[i % len(q_in)].put((i, item))

                # 任务分派完成之后，再为每个输入进程分配一个None
                for q in q_in:
                    q.put(None)

                # 等待每个输入进程结束
                for p in read_process:
                    p.join()

                # 给输出队列放入一个None，代表结束
                q_out.put(None)

                # 等待输出进程（写进程）结束
                write_process.join()

        if not count:
            print('Did not find and list file with prefix %s'%args.prefix)