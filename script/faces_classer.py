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
import numpy as  np
import sklearn
import shutil
from mxnet import ndarray as nd
np.set_printoptions(suppress=True)
from sklearn.cluster import DBSCAN

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

class Model():
    # 模型加载
    def __init__(self,image_size, args):
        prefix, epoch = args.model.split(',')
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        self.model = mx.mod.Module(symbol=sym, context=mx.gpu())
        self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        self.model.set_params(arg_params, aux_params)

    # 得到输入图像的特征向量
    def predict(self,img):
        img = nd.array(img)
        #print(img.shape)
        img = nd.transpose(img, axes=(2, 0, 1)).astype('float32')
        img = nd.expand_dims(img, axis=0)
        #print(img.shape)
        db = mx.io.DataBatch(data=(img,))

        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        embedding = net_out[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding,axis=1)
        return embedding



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
            #print(os.path.relpath(k, root), v)
            pass
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
    print('生成文件：%s'%path_out)
def make_list(args):
    """Generates .lst file.
    Parameters
    ----------
    args: object that contains all the arguments
    """
    image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)

    path = os.path.join(args.root,'.lst')
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


def get_outdir_emds(args):
    dir_list = os.listdir(args.outdir)
    if 'embeddings.npz' in dir_list:
        path = os.path.join(args.outdir,'embeddings.npz')
        data =  np.load(path)
        npz_embs = data['npz_embs']
        npz_emb_len = data['npz_emb_len']
        return npz_embs,npz_emb_len
    return None,None


def gen_npz_embs(q_out,model):

    buf = {}
    more = True # 队列处理完成，设置为None
    pre_id = 0 # 记录前一张图片的ID，如果前一张，和当前的ID不同，表示前一个人的所有图片已经预测完成
    count = 0

    # 每个id的特征向量
    id_embeddings = None

    # 存入ngs.npz中embeddi，所有ID的平均特征向量
    npz_embs = None

    # 存入ngs.npz中embeddi，每个特征向量对应的图片张数
    npz_emb_len = None

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
            print(i)

        # 如果为空，说明已经预测完了所有图片
        else:
            # 计算最后一个人的平均特征向量
            id_mean_emb = np.mean(id_embeddings, axis=0)
            # 为了统一，扩展一个维度
            id_mean_emb = np.expand_dims(id_mean_emb, axis=0)
            # 垂直方向进行拼接
            npz_embs = np.vstack((npz_embs, id_mean_emb))

            # 保存平均向量对应ID图片的数目
            id_mean_embs_len = id_embeddings.shape[0]
            # 为了统一，扩展一个维度
            id_mean_embs_len = np.expand_dims(id_mean_embs_len, axis=0)

            # 垂直方向进行拼接
            npz_emb_len = np.vstack((npz_emb_len, id_mean_embs_len))
            more = False

        while count in buf:
            img, item = buf[count]
            #print(img.shape ,item)
            del buf[count]
            if img is not None:
                # 对一张图片进行特征向量预测
                embedding = model.predict(img)

                # 判断ID的所有图片是否全部预测完成
                if (int(pre_id) != int(item[2])):

                    # 对单个ID的所有特征向量求平均值
                    id_mean_emb = np.mean(id_embeddings, axis=0)
                    # 为了统一增加一个维度
                    id_mean_emb = np.expand_dims(id_mean_emb, axis=0)

                    # 记录单个ID其有多少张图片
                    id_mean_embs_len = id_embeddings.shape[0]
                    # 为了统一增加一个维度
                    id_mean_embs_len = np.expand_dims(id_mean_embs_len, axis=0)

                    # 如果为None，说明这是第一个ID的平均特征向量，直接赋值即可
                    if npz_embs is None:
                        npz_embs = id_mean_emb
                        npz_emb_len = id_mean_embs_len
                    # 如果不是第一个，则进行锤子凭借
                    else:
                        npz_embs = np.vstack((npz_embs, id_mean_emb))
                        npz_emb_len = np.vstack((npz_emb_len, id_mean_embs_len))

                    # 把记录单个ID所有的id_embeddings设置为Nobe
                    id_embeddings = None

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

    # 返回计算完成所有的特征向量，以及特征向量对应的图片数目
    return npz_embs,npz_emb_len



def get_cos_simi(npz_embs,embedding):
    mol = np.matmul(npz_embs, embedding.T).reshape(-1)
    denom = np.linalg.norm(npz_embs, axis=1) * np.linalg.norm(embedding, axis=1)
    cos = mol / denom.reshape(-1)
    sim = 0.5 + 0.5 * cos
    return sim


def face_in_storage(args, npz_embs, npz_emb_len, idx, embedding, item, fd, max_nums):
    # 得到保存图像的目录
    lib_dir_path = os.path.join(args.outdir, '%08d' % idx)
    # 原图像的路径
    old_img_path = os.path.join(args.indir, item[1])
    # 新图片的路径
    new_img_path = os.path.join(lib_dir_path, '%05d' % (npz_emb_len[idx]) + args.encoding)
    # 进行复制

    fd.write(old_img_path + '\t' + new_img_path + '\t' + str(max_nums) + '\n\n')
    shutil.copyfile(old_img_path, new_img_path)

    if args.delete:
        os.remove(old_img_path)

    # 对特征向量，已经对应的长度进行更新
    old_len = npz_emb_len[idx][0]
    new_len = old_len + 1
    npz_embs[idx] = (npz_embs[idx].reshape(1, -1) * old_len + embedding) / new_len
    npz_emb_len[idx] = new_len
    return npz_embs, npz_emb_len

def face_create_lib(args, npz_embs, npz_emb_len, embedding, item, fd, max_nums):
# 得到当前ID的总个数
    id_sum = npz_embs.shape[0]

    # 为新的人脸入库，创建一个新的文件夹
    new_lib_dir = os.path.join(args.outdir, '%08d' % id_sum)
    os.mkdir(new_lib_dir)

    # 为了统一，扩展一个维度
    embedding = np.expand_dims(embedding, axis=0)

    # 特征向量以及对应的ID图片的数目，都进行垂直拼接
    npz_embs = np.vstack((npz_embs, embedding.reshape(1, -1)))
    npz_emb_len = np.vstack((npz_emb_len, np.array([[1]])))

    new_img_path = os.path.join(new_lib_dir, '00000' + args.encoding)
    old_img_path = os.path.join(args.indir, item[1])

    fd.write(old_img_path + '\t' + new_img_path + '\t' + str(max_nums) + '\n\n')
    shutil.copyfile(old_img_path, new_img_path)

    if args.delete:
        os.remove(old_img_path)
    return npz_embs,npz_emb_len

def face_not_recog(args, item, num, fd, max_nums):
    # 为新的人脸入库，创建一个新的文件夹
    recog_dir = os.path.join(args.outdir, 'not_recognition')
    if not os.path.exists(recog_dir):
        os.mkdir(recog_dir)

    #     # 原图像的路径
    old_img_path = os.path.join(args.indir, item[1])
    new_img_path = os.path.join(recog_dir, '%08d' % num + args.encoding)

    fd.write(old_img_path + '\t' + new_img_path + '\t' + str(max_nums) + '\n\n')
    shutil.copyfile(old_img_path, new_img_path)

    if args.delete:
        os.remove(old_img_path)


def write_worker(q_out, fname, args):
    """Function that will be spawned to fetch processed image
    from the output queue and write to the .rec file.
    Parameters
    ----------
    q_out: queue
    """

    # 尝试加载embeddings.npz文件，该文件主要存在两个数组
    # 'npz_embs':保存每ID的平均向量， 'npz_emb_len':每个ID存在多少图片，他们是一一对应的关系
    npz_embs,npz_emb_len = get_outdir_emds(args)

    # 模型加载
    model = Model([112, 112],args)

    # 保存embeddings.npy的路径
    path = os.path.join(args.outdir, 'embeddings')

    # 如果没有embeddings.npz文件，则为输出目录创建embeddings.npz文件
    if npz_embs is None:
        npz_embs,npz_emb_len = gen_npz_embs(q_out,model)


    else:
        buf = {}
        more = True
        count = 0
        pre_time = time.time()
        not_recog_conut = 0
        fd_in_storage = open(os.path.join(args.indir,'in_storage.txt'),'w')
        fd_not_recog = open(os.path.join(args.indir, 'not_recog.txt'),'w')
        fd_create_lib = open(os.path.join(args.indir, 'create_lib.txt'),'w')
        while more:
            deq = q_out.get()
            if deq is not None:
                i, img, item = deq
                # print(i)
                buf[i] = (img, item)
            else:
                #print(npz_embs.shape)
                more = False

            while count in buf:
                img, item = buf[count]

                # print(item)
                del buf[count]
                if img is not None:

                    embedding = model.predict(img)

                    # 求得需要预测的特征向量，与保存的每个特征向量的余弦相似度
                    simi = get_cos_simi(npz_embs,embedding)

                    # 找到最相似的图像
                    idx = np.argmax(simi)
                    idx_min = np.argmin(simi)

                    # 如果大于阈值，说明人脸库中，已经有了相似的人脸,进行人脸入库

                    max_nums = simi.argsort()[-5:][::-1]
                    if simi[idx] > args.max_threshold:
                        npz_embs, npz_emb_len = face_in_storage(args, npz_embs, npz_emb_len, idx, embedding, item, fd_in_storage, max_nums)
                        pass
                    elif simi[idx] > args.min_threshol:

                        face_not_recog(args, item, not_recog_conut, fd_not_recog, max_nums)
                        not_recog_conut += 1
                        pass
                    else:
                        # 人脸库不存在人脸,则创建一个新的文件夹，入库
                        npz_embs, npz_emb_len = face_create_lib(args, npz_embs, npz_emb_len, embedding, item, fd_create_lib, max_nums)
                        #print(simi[idx])
                        #print('请核实文件，认为其为非人脸： %s' %item[1])
                        pass

                if count % 100 == 0:
                    cur_time = time.time()
                    print('time:', cur_time - pre_time, ' count:', count)
                    pre_time = cur_time
                count += 1
        fd_in_storage.close()
        fd_not_recog.close()
        fd_create_lib.close()
    # 保存npz文件
    np.savez(path, npz_embs=npz_embs, npz_emb_len=npz_emb_len)


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

    # 想要生成.lst的目录,再人脸为空的情况下，必须提前为输入以及输出文件夹生成一个.lst文件
    parser.add_argument('--root', default='F:/3.1no_recognition' ,help='path to folder containing images.')

    # 需要进行人脸分类的文件夹
    parser.add_argument('--indir', default='F:/3.1no_recognition' ,help='path to folder containing images.')

    # 人脸分类之后输出的文件夹，注意，该问价夹中，必须存在一个ID，也就是说，至少存在一个子文件夹，并且该子文件夹中最少包含一张照片
    parser.add_argument('--outdir', default='F:/3.2image_classer' ,help='path to folder containing images.')

    # 预训练模型的路径
    parser.add_argument('--model', default='../models/my/model-y1-test2/model,13', help='path to load model.')
    #parser.add_argument('--model', default='../models/model-y1-test2/model,0', help='path to load model.')

    # 判定是否为同一人的相似度阈值
    parser.add_argument('--max_threshold', default=0.75, type=float, help='')

    # 低于该阈值，确定为一个新的人脸
    parser.add_argument('--min_threshol', default=0.665, type=float, help='')

    cgroup = parser.add_argument_group('Options for creating image lists')


    # 该处填写为True，则代表是为root对应的目录下生成.lst文件，注意，此时不会进行人脸分类
    cgroup.add_argument('--list', default=False,help='')

    cgroup.add_argument('--no-shuffle', dest='shuffle', action='store_true',
                        help='If this is passed, \
        im2rec will not randomize the image order in <prefix>.lst')

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

    # 是否删除输入文件夹的文件
    rgroup.add_argument('--delete', default=True, type=bool, help='')

    # 读取图片，进行解码的线程数目
    rgroup.add_argument('--num-thread', type=int, default=3,
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
    args.root = os.path.abspath(args.root)
    npz_path = os.path.join(args.outdir,'embeddings.npz')

    # 如果输出目录不存在embeddings文件，则为输出目录先生成embeddings文件
    if not os.path.exists(npz_path):
        args.indir =args.outdir
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
                write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, args))
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