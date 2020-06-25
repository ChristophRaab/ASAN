import argparse
import os
import os.path as osp
import sys
sys.path.append("C:/Users/craab/Desktop/owncloud/transfer_learning")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
import pandas as pd
import re
from train_image import train,make_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='data/amazon.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/webcam.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--sn', type=bool, default=False, help="whether to use spectral normalization")
    parser.add_argument('--tl', type=str, default="RSL", help="transfer_loss")
    parser.add_argument('--k', type=int, default=1, help="K Parameter of RSL")
    parser.add_argument('--tllr', type=float, default=0.0001, help="transfer_loss learning rate")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loader workers")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    available_datasets = {
        'w':'data/webcam.txt',
        'd':'data/dslr.txt',
        'a':'data/amazon.txt',
            }
    shrinkage_parameters = [1,4,7,11,15]
    tl_lrs = [0.001,0.0001,0.00001]
    results = []
    for k in shrinkage_parameters:
        for tllr in tl_lrs:
            tmp_acc = []
            for i in range(3):

                # train config
                args.k = k
                args.tllr = tllr
                # config = make_config(args)
                # acc = train(config)
                acc = 1
                tmp_acc.append(acc)

            results.extend([k,tllr,np.mean(tmp_acc),np.std(tmp_acc)])
    results = np.reshape(results,[-1,4])[9]
    results = pd.DataFrame(results)
    results.to_csv("optimize_"+re.split(r'[.,/]', args.s_dset_path)[1]+"_"+re.split(r'[.,/]', args.t_dset_path)[1]+"_"+args.tl+"_"+str(args.sn)+".csv")

#optimize_image.py --tl RSL --sn True --s_dset_path data/amazon.txt --t_dset_path data/webcam.txt
#optimize_image.py --tl RSL --sn True --s_dset_path data/dslr.txt --t_dset_path data/webcam.txt
#optimize_image.py --tl RSL --sn True --s_dset_path data/webcam.txt --t_dset_path data/dslr.txt
#optimize_image.py --tl RSL --sn True --s_dset_path data/amazon.txt --t_dset_path data/dslr.txt
#optimize_image.py --tl RSL --sn True --s_dset_path data/dslr.txt --t_dset_path data/amazon.txt
#optimize_image.py --tl RSL --sn True --s_dset_path data/webcam.txt --t_dset_path data/amazon.txt