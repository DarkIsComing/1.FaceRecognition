
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
import traceback
import shutil

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

def dirs_rename(args):

    newdir_idx = args.stdir_idx
    for old_path, dirs, files in os.walk(args.root, followlinks=True):
        dirs.sort()
        files.sort()
        if old_path == args.root:
            continue

        if (args.delete_mixnums != 0)   and (len(files) <=args.delete_mixnums):
            shutil.rmtree(old_path)
            continue

        if (len(files))==0:
            print(old_path)

        num_file =0
        for old_file in files:
            if old_file[-4:] == args.img_suffix:
                new_file = '%05d%s' %(num_file,args.img_suffix)
                old_fil_path = os.path.join(old_path,old_file)
                new_fil_path = os.path.join(old_path, new_file)
               # print(old_fil_path)
                #print(new_fil_path)
                os.rename(old_fil_path,new_fil_path)
                num_file += 1

        old_path_prefix =  old_path.replace('\\','/').split('/')
        old_path_prefix = '/'.join(old_path_prefix[:-1])

        new_path_suffix = '%08d' %(newdir_idx)

        new_path = os.path.join(old_path_prefix,new_path_suffix)
        #print('old_path',old_path + '/')
        #print('new_path',new_path + '/')
        os.rename(old_path, new_path)

        newdir_idx += 1



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



def dataset_divi(args):
    rootdir_list = os.listdir(args.root)
    idsdir_list = [name for name in rootdir_list if os.path.isdir(os.path.join(args.root, name))]
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(idsdir_list)
    idsdir_len = len(idsdir_list)
    train_len = int(idsdir_len * args.train_ratio)
    train_list = idsdir_list[:train_len]
    test_list = idsdir_list[train_len:]
    print('train_len: ',train_len)
    print('test_len: ',len(test_list))



    for dir_name in train_list:
        old_dir = os.path.join(args.root, dir_name )
        new_dir = os.path.join(args.outdir, os.path.join('train',dir_name))
        shutil.copytree(old_dir,new_dir)

    for dir_name in test_list:
        old_dir = os.path.join(args.root, dir_name)
        new_dir = os.path.join(args.outdir, os.path.join('test',dir_name))
        shutil.copytree(old_dir,new_dir)



def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='数据预处理的一些工具')

    # 如果设置为0代表目录进行规则化，如果设置为1代表进行数据集的分割
    parser.add_argument('--rule_divi', type=int, default=0, choices=[0,1],
                        help='specify the encoding of the images.')

    # 需要规则化，或者需要进行数据集分割的根目录s
    #parser.add_argument('--root',default='/data/zwh/1.FaceRecognition/2.Dataset/2.PaidOnData/2.DataDivi/1.Shunde/5.dataset_divi/pack3/train',help='')
    parser.add_argument('--root',default='/data/zwh/1.FaceRecognition/2.Dataset/2.PaidOnData/2.DataDivi/1.Shunde/3.2image_classer',help='')

    rule = parser.add_argument_group('把目录下的文件按照一定规则命名，相关的参数')

    rule.add_argument('--stdir_idx', default=0, type=int, help = '文件夹序列的起始数字')
    rule.add_argument('--img_suffix', type=str, default='.jpg', choices=['.jpg', '.png','.bmp'],
                        help='specify the encoding of the images.')

    # 设置为0代表不删除
    rule.add_argument('--delete_mixnums', default=0, type=int, help = '伤处图片张数小于或者等于设定值得文件夹')


    divi = parser.add_argument_group('划分数据训练集，测试集，以及验证测试集')

    # 数据划分之后输出的目录
    divi.add_argument('--outdir',default='/data/zwh/1.FaceRecognition/2.Dataset/2.PaidOnData/2.DataDivi/1.Shunde/5.dataset_divi/pack3',help='')

    # 训练集所占用的比例
    divi.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of images to use for training.')

    # 是否随机进行分割
    divi.add_argument('--no-shuffle', dest='shuffle', action='store_false',default=True,
                        help='If this is passed, \
        im2rec will not randomize the image order in <prefix>.lst')



    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.rule_divi == 0:
        dirs_rename(args)
    elif args.rule_divi == 1:
        dataset_divi(args)

    pass