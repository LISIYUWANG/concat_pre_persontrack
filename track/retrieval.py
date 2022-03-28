# -*- coding: utf-8 -*-
# @Time   : 2022/3/28 13:43
# @Author : LWDZ
# @File   : retrieval.py
# @aim    : image retrieval
#--------------------------------------------------

# -*- coding: utf-8 -*-
'''
图像搜索：单张图，一种图向量
'''
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# 使用第一张与第2张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import argparse
from track.get_features import *
import PIL.Image as img
from pathlib import Path
import glob
import re
import pickle
#建立新文件夹
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    # if not dir.exists() and mkdir:
    #     print('新路径')
    #     dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path
#图片转存
def img_trans(infile_path,outfile_folder,similarity,flag=True):
    im = img.open(infile_path)
    name = infile_path.split('/')[-1]
    if flag == False:
        similarity=str(similarity).split('.')[1]
    else:
        similarity = str(similarity)
    outfile = str(outfile_folder)+'/'+str(name)+'_'+similarity
    print('out ',outfile)
    im.save(outfile)
# sort the images
def sort_img(qf, gf):
    score = gf*qf
    score = score.sum(1)
    # predict index
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    import numpy as np
    s = np.around(s, 3)
    return s, index

def main_worker(args):

    parser.add_argument('--model_path',  default='./models/net_best.pth', help="path of feature extracting models")
    parser.add_argument('--gallery_feature_path',default='../features/data', help="gallery feature path")
    parser.add_argument('--gallery_feature_label_path', default='../features/data', help="gallery feature folder path")
    parser.add_argument('--query_img_folder', default='../dataset/query', help="query images path")
    parser.add_argument('--origin_imgs_path', default='../dataset/origin_data', help="original images path")
    parser.add_argument('--use_gpu', default=True, help="if use gpu ")

    model_path = args.model_path
    gallery_feature_path = args.gallery_feature_path
    gallery_feature_label_path = args.gallery_feature_label_path
    query_img_folder = args.query_img_folder
    query_result_folder = args.query_result_folder
    origin_imgs_path = args.origin_imgs_path   # 保存显示的未处理的png格式的原图
    use_gpu = args.use_gpu

    # Prepare data.

    # get gallery features.
    gallery_feature_label = pickle.load(open(gallery_feature_label_path, 'rb'))
    gallery_feature = pickle.load(open(gallery_feature_path, 'rb'))
    print(np.array(gallery_feature).shape)

    '''264212446 26421122  264212517 264212532 264213630 26421193 26421422'''
    # Query.
    # Extract query features.
    query_result_folder = increment_path(query_result_folder, exist_ok=False)  # increment run
    for img_name in os.listdir(query_img_folder):
        img_path = os.path.join(query_img_folder,img_name)
        query_feature = extract_feature_query(img_path, model_path)
        # Sort.
        similarity, index = sort_img(query_feature, gallery_feature)
        #similar_dir = 'similar/top'
        sorted_paths = [gallery_feature_label[i] for i in index]
        img_name2 = img_name.split('/')[-1]
        query_result_folder2=query_result_folder+'/'+img_name2
        query_result_folder2 = increment_path(query_result_folder2, exist_ok=False)  # increment run
        img_trans(img_path, query_result_folder2, 'origin', flag=True)

        for i in range(20):
            pth = sorted_paths[i].split('/')[-1]
            pth = origin_imgs_path +'/'+pth
            print(pth)
            print(sorted_paths[i])  # 输出top10
            img_trans(pth, query_result_folder2, similarity[i],False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="retrieval images")
    # data
    parser.add_argument('--model_path',  default='./models/net_best.pth', help="path of feature extracting models")
    parser.add_argument('--gallery_feature_path',default='../features/data', help="gallery feature path")
    parser.add_argument('--gallery_feature_label_path', default='../features/data', help="gallery feature folder path")
    parser.add_argument('--query_img_folder', default='../dataset/query', help="query images path")
    parser.add_argument('--query_result_folder', default='../result/query', help="the result of query images path")
    parser.add_argument('--origin_imgs_path', default='../dataset/origin_data', help="original images path")
    parser.add_argument('--use_gpu', default=True, help="if use gpu ")
    args = parser.parse_args()

    '''
    origin_img_path
    query_img_path
    model_path
    gallery_feature_path
    gallery_feature_label_path
    '''
    main_worker(args)


