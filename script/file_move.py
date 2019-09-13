
import random
import argparse
import cv2
import time
import shutil
import os
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

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('--indir',
                        default='F:/sunde/00000000',
                        help='path to folder containing images.')

    parser.add_argument('--outdir',
                        default='F:/sunde/00000002',
                        help='path to folder containing images.')


    # 要移动图片的其实下标
    parser.add_argument('--move_start_idx', default=200001, type=int,
                        help='')


    # 要移动文件的数目
    parser.add_argument('--move_nums', default=100000, type=int,
                        help='')

    # 需要分类图片的格式
    parser.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png','.bmp'],
                        help='list of acceptable image extensions.')

    # 是否进行递归搜索
    parser.add_argument('--recursive', default=True,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')

    return  parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    move_idx =  args.move_start_idx
    for i in range(args.move_nums):
        odl_path = os.path.join(args.indir, '%08d' %move_idx +'.jpg')
        new_path = os.path.join(args.outdir, '%08d' %move_idx +'.jpg')
        shutil.move(odl_path, new_path)
        move_idx += 1