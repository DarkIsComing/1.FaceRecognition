#-*- coding: utf-8 -*-
import numpy as np
import os
from easydict import EasyDict as edict

# config配置是最基本的配置，如果后面出现相同的，则被覆盖
config = edict()

config.bn_mom = 0.9 # 反向传播的momentum
config.workspace = 256 # mxnet需要的缓冲空间
config.emb_size = 128 #  输出特征向量的维度
config.ckpt_embedding = False # 是否检测输出的特征向量
config.net_se = 0 # 暂时不知道
config.net_act = 'prelu' # 激活函数
config.net_unit = 3 #
config.net_input = 1 #
config.net_blocks = [1,4,6,2]
config.net_output = 'E' # 输出层，链接层的类型，如"GDC"也是其中一种，具体查看recognition\symbol\symbol_utils.py
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30'] # 测试数据，即.bin为后缀的文件
config.ce_loss = True #Focal loss，一种改进的交叉损失熵
config.fc7_lr_mult = 1.0 # 学习率的倍数
config.fc7_wd_mult = 1.0 # 权重刷衰减的倍数
config.fc7_no_bias = False #
config.max_steps = 0 # 训练的最大步骤吧，感觉有点懵逼，不过不影响大局
config.data_rand_mirror = True # 数据随机进行镜像翻转
config.data_cutoff = False # 数据进行随机裁剪
config.data_color = 0 # 估计是数据进行彩色增强
config.data_images_filter = 0 #表示每个人的图像数目要大于该值才进行训练
config.count_flops = True # 是否计算一个网络占用的浮点数内存
config.memonger = False #not work now



# 可以看到雨哦很多网络结构，就不为大家一一注释了
# 以为我也没有把每个网络都弄得很透彻，可以看到又很多网络结构，在训练的时候我们都是可以选择的
# r100 r100fc
# network settings r50 r50v1 d169 d201 y1 m1 m05 mnas mnas025
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r100fc = edict()
network.r100fc.net_name = 'fresnet'
network.r100fc.num_layers = 100
network.r100fc.net_output = 'FC'

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50v1 = edict()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1

network.d169 = edict()
network.d169.net_name = 'fdensenet'
network.d169.num_layers = 169
network.d169.per_batch_size = 64
network.d169.densenet_dropout = 0.0

network.d201 = edict()
network.d201.net_name = 'fdensenet'
network.d201.num_layers = 201
network.d201.per_batch_size = 64
network.d201.densenet_dropout = 0.0

network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'

network.y2 = edict()
network.y2.net_name = 'fmobilefacenet'
network.y2.emb_size = 256
network.y2.net_output = 'GDC'
network.y2.net_blocks = [2,8,16,4]

network.m1 = edict()
network.m1.net_name = 'fmobilenet'
network.m1.emb_size = 256
network.m1.net_output = 'GDC'
network.m1.net_multiplier = 1.0

network.m05 = edict()
network.m05.net_name = 'fmobilenet'
network.m05.emb_size = 256
network.m05.net_output = 'GDC'
network.m05.net_multiplier = 0.5

network.mnas = edict()
network.mnas.net_name = 'fmnasnet'
network.mnas.emb_size = 256
network.mnas.net_output = 'GDC'
network.mnas.net_multiplier = 1.0

network.mnas05 = edict()
network.mnas05.net_name = 'fmnasnet'
network.mnas05.emb_size = 256
network.mnas05.net_output = 'GDC'
network.mnas05.net_multiplier = 0.5

network.mnas025 = edict()
network.mnas025.net_name = 'fmnasnet'
network.mnas025.emb_size = 256
network.mnas025.net_output = 'GDC'
network.mnas025.net_multiplier = 0.25



# 可以看到存在emore与retina两个数据集，训练的时候我们只能指定一个。
# num_classes来自property，为人脸id数目，为了能够较好的拟合数据
# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = '/data/zwh/1.FaceRecognition/2.Dataset/2.PaidOnData/1.TrainData/pack3'
dataset.emore.num_classes = 623
dataset.emore.image_shape = (112,112,3)
#dataset.emore.val_targets = ['lfw','cfp_ff','cfp_fp', 'agedb_30','shunde']
dataset.emore.val_targets = ['lfw','shunde']

dataset.retina = edict()
dataset.retina.dataset = 'retina'
dataset.retina.dataset_path = '../datasets/ms1m-retinaface-t1'
dataset.retina.num_classes = 502
dataset.retina.image_shape = (112,112,3)
dataset.retina.val_targets = ['lfw', 'cfp_fp', 'agedb_30']


# 损失函数是我们的重点，大家看了之后，不要觉得太复杂，
# loss_m1，loss_m2，loss_m3，其出现3个m，作者是为了减少代码量，把多个损失函数合并在一起了
# 即nsoftmax，arcface，cosface，combined
loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'

loss.nsoftmax = edict()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.triplet = edict()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 5
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 7200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 60
loss.triplet.lr = 0.05

loss.atriplet = edict()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35
loss.atriplet.triplet_bag_size = 7200
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

# default settings
default = edict()

# default network
default.network = 'y1'
#default.pretrained = ''
default.pretrained = '../models/my/model-y1-test2/model'
default.pretrained_epoch = 9
# default dataset
default.dataset = 'emore'
default.loss = 'arcface'
default.frequent = 20 # 每20个批次打印一次准确率等log
default.verbose = 200 # 每训练2000次，对验证数据进行一次评估
default.kvstore = 'device' #键值存储

default.end_epoch = 100000 # 结束的epoch
default.lr = 0.000001 # 初始学习率，如果每个批次训练的数目小，学习率也相应的降低
default.wd = 0.0005 # 大概是权重初始化波动的范围
default.mom = 0.9
default.per_batch_size = 240 # 每存在一个GPU，训练48个批次，如两个GPU，则实际训练的batch_size为96
default.ckpt = -1 # 该设置为2，每次评估的时候都会保存模型
#default.lr_steps = '10000,160000,220000'  # 每达到步数，学习率变为原来的百分之十
default.lr_steps = '20000,50000,100000'  # 每达到步数，学习率变为原来的百分之十
default.models_root = './models' # 模型保存的位置


# 对config = edict()进行更新
def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
      config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

