# -*- coding: utf-8 -*-
# Citation:
# @article{qu2021transmef,
#   title={TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning},
#   author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
#   journal={arXiv preprint arXiv:2112.01030},
#   year={2021}
# }
import torch
import argparse
from utils import test
from collections import OrderedDict
from Network_TransMEF import TransNet
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--model_path', type=str, default='./bestmodel.pth', help='best model path')
parser.add_argument('--test_path', type=str, default='./MEFB_dataset/', help='test dataset path')
parser.add_argument('--result_path', type=str, default='./TransMEF_result', help='test result path')
args = parser.parse_args()
device = 'cuda'

model = TransNet().to(device)

state_dict = torch.load(args.model_path, map_location='cuda:0')['model']

if len(args.gpus) > 1:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)

model.eval()

test(args.test_path, model, save_path=args.result_path)
