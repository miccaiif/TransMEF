# -*- coding: utf-8 -*-
# Citation:
# @article{qu2021transmef,
#   title={TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning},
#   author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
#   journal={arXiv preprint arXiv:2112.01030},
#   year={2021}
# }

import string
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from glob import glob
import os
from PIL import Image, ImageFile
import torch
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True
_tensor = transforms.ToTensor()
_pil_gray = transforms.ToPILImage()
device = 'cuda'


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def load_img(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    return _tensor(img).unsqueeze(0)


class Strategy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y1, y2):
        return (y1 + y2) / 2


def read_image(path):
    I = np.array(Image.open(path))
    return I


def fusion(x1, x2, model):
    with torch.no_grad():
        start = time.time()
        fusion_layer = Strategy().to(device)
        feature1 = model.encoder(x1)
        feature2 = model.encoder(x2)
        feature_fusion = fusion_layer(feature1, feature2)
        out = model.decoder(feature_fusion).squeeze(0).detach().cpu()
        time_used = time.time() - start
        print("fusion time：", time_used, " used")
        return out


class Test:
    def __init__(self):
        pass

    def load_imgs(self, img1_path, img2_path, device):
        img1 = load_img(img1_path).to(device)
        img2 = load_img(img2_path).to(device)
        return img1, img2

    def save_imgs(self, save_path, save_name, img_fusion):
        mkdir(save_path)
        save_path = os.path.join(save_path, save_name)
        img_fusion.save(save_path)


class test_gray(Test):
    def __init__(self):
        super().__init__()
        self.img_type = 'gray'

    def get_fusion(self, img1_path, img2_path, model,
                   save_path='none', save_name='none'):
        img1, img2 = self.load_imgs(img1_path, img2_path, device)
        img_fusion = fusion(x1=img1, x2=img2, model=model)
        img_fusion = MaxMinNormalization(img_fusion[0], torch.max(img_fusion[0]), torch.min(img_fusion[0]))
        img_fusion = _pil_gray(img_fusion)

        self.save_imgs(save_path, save_name, img_fusion)
        return img_fusion


def test(test_path, model, save_path='./test_result/'):
    img_list = glob(test_path + '*')
    img_num = len(img_list) / 2
    suffix = img_list[0].split('.')[-1]
    img_name_list = list(
        set([img_list[i].split('\\')[-1].split('.')[0].strip(string.digits) for i in range(len(img_list))]))

    fusion_phase = test_gray()

    for i in range(int(img_num)):
        img1_path = img_name_list[0] + str(i) + '.' + suffix
        img2_path = img_name_list[1] + str(i) + '.' + suffix
        save_name = 'fusion_' + str(i) + '.' + suffix
        fusion_phase.get_fusion(img1_path, img2_path, model,
                                save_path=save_path, save_name=save_name)
