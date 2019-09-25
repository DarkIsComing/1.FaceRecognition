import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))

parser = argparse.ArgumentParser(description='Package LFW images')
# general
parser.add_argument('--data-dir', default='/data/zwh/1.FaceRecognition/2.Dataset/2.PaidOnData/2.DataDivi/1.Shunde/5.dataset_divi/pack3/test', help='')
parser.add_argument('--image-size', type=str, default='112,112', help='')
parser.add_argument('--output', default='/data/zwh/1.FaceRecognition/2.Dataset/2.PaidOnData/2.DataDivi/1.Shunde/5.dataset_divi/pack3/test/shunde.bin', help='path to save.')
args = parser.parse_args()
lfw_dir = args.data_dir
image_size = [int(x) for x in args.image_size.split(',')]


def read_pairs(pairs_filename):
  pairs = []
  with open(pairs_filename, 'r') as f:
    for line in f.readlines():
      pair = line.strip().split()
      pairs.append(pair)
  return np.array(pairs)

def get_paths(data_dir, pairs, file_ext):
  nrof_skipped_pairs = 0
  path_list = []
  issame_list = []
  for pair in pairs:
    if len(pair) == 3:
      path0 = os.path.join(data_dir, pair[0])
      path1 = os.path.join(data_dir, pair[1])
      if int(pair[2]) == 1:
        issame = True
      else:
        issame = False
    elif len(pair) == 4:
      path0 = os.path.join(data_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
      path1 = os.path.join(data_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
      issame = False
    if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
      path_list += (path0, path1)
      issame_list.append(issame)
    else:
      print('not exists', path0, path1)
      nrof_skipped_pairs += 1
  if nrof_skipped_pairs > 0:
    print('Skipped %d image pairs' % nrof_skipped_pairs)

  return path_list, issame_list





data_pairs = read_pairs(os.path.join(lfw_dir, 'pairs.txt'))
data_paths, issame_list = get_paths(lfw_dir, data_pairs, 'jpg')
print(len(data_paths))
print(len(issame_list))

lfw_bins = []
#lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
i = 0
for path in data_paths:
  with open(path, 'rb') as fin:
    _bin = fin.read()
    lfw_bins.append(_bin)
    #img = mx.image.imdecode(_bin)
    #img = nd.transpose(img, axes=(2, 0, 1))
    #lfw_data[i][:] = img
    i+=1
    if i%1000==0:
      print('loading data', i)

with open(args.output, 'wb') as f:
  pickle.dump((lfw_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
