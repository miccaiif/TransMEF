# -*- coding: utf-8 -*-
# Citation:
# @inproceedings{qu2022transmef,
#   title={Transmef: A transformer-based multi-exposure image fusion framework using self-supervised multi-task learning},
#   author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
#   booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
#   volume={36},
#   number={2},
#   pages={2126--2134},
#   year={2022}
# }

import cv2
import argparse
from collections import OrderedDict
from Network_TransMEF import TransNet
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
_pil_rgb = transforms.ToPILImage('RGB')
_pil_gray = transforms.ToPILImage()
device = 'cuda'


def get_block(img, block_size=256):
    '''
    The original image is cut into blocks according to block_size
    output: blocks [blocks_num, block_size, block_size]
    '''
    blocks = []
    m, n = img.shape

    img_pad = np.pad(img, ((0, 256 - m % block_size), (0, 256 - n % block_size)), 'reflect')  # mirror padding
    m_block = int(np.ceil(m / block_size))  # Calculate the total number of blocks
    n_block = int(np.ceil(n / block_size))  # Calculate the total number of blocks

    # cutting
    for i in range(0, m_block):
        for j in range(0, n_block):
            block = img_pad[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
            blocks.append(block)
    blocks = np.array(blocks)
    return blocks


def fuse(img1, img2):
    '''
    block fusion
    '''
    block_num = img1.shape[0]

    final_fusion = np.zeros_like(img1)

    for i in range(block_num):
        img1_inblock = _tensor(img1[i, :, :]).unsqueeze(0).to(device)
        img2_inblock = _tensor(img2[i, :, :]).unsqueeze(0).to(device)

        img_fusion = fusion(x1=img1_inblock, x2=img2_inblock, model=model)

        # note that no normalization should be used in different block fusion
        # img_fusion = MaxMinNormalization(img_fusion[0], torch.max(img_fusion[0]), torch.min(img_fusion[0]))
        # img_fusion = img_fusion.numpy()
        img_fusion = _pil_gray(img_fusion)
        img_fusion = np.asarray(img_fusion)

        final_fusion[i,:,:] = img_fusion

    return final_fusion


def block_to_img(block_img, m, n):
    '''
    Enter the fused block and restore it to the original image size.
    '''
    block_size = block_img.shape[2]
    m_block = int(np.ceil(m / block_size))
    n_block = int(np.ceil(n / block_size))
    fused_full_img_wpad = np.zeros((m_block * 256, n_block * 256), dtype=int)  # Image size after padding
    for i in range(0, m_block):
        for j in range(0, n_block):
            fused_full_img_wpad[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size] = block_img[i * n_block + j, :, :]
        fused_full_img = fused_full_img_wpad[:m, :n]  # image with original size
    return fused_full_img


def block_fusion(img1, img2, block_size=256):
    '''
    Input img1, img2, slice block according to block_size and fuse, output result
    '''
    # blocks_img大小[blocks_num, block_size, block_size, 3]
    blocks_img1 = get_block(img1, block_size=block_size)
    blocks_img2 = get_block(img2, block_size=block_size)
    print('img1', blocks_img1.shape)
    print('img2', blocks_img2.shape)

    # fusion
    fused_block_img1 = fuse(blocks_img1, blocks_img2)

    # block restore to orginal size
    fused_img = block_to_img(fused_block_img1, img1.shape[0], img1.shape[1])

    # visualization
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.title('ori_img1')
    # plt.imshow(img1,cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.title('ori_img2')
    # plt.imshow(img2,cmap='gray')
    # plt.subplot(2, 2, 3)
    # plt.title('fused_img')
    # plt.imshow(fused_img,cmap='gray')
    # plt.savefig('./test.jpg')

    return fused_img


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


def load_img_cv(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    return img


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
        img1 = load_img_cv(img1_path)
        img2 = load_img_cv(img2_path)
        return img1, img2

    def save_imgs(self, save_path, save_name, img_fusion):
        mkdir(save_path)
        save_path = os.path.join(save_path, save_name)
        # from matplotlib import cm
        # img_fusion = Image.fromarray(np.uint8(cm.gist_earth(img_fusion) * 255))
        # cv2.imwrite(save_path,img_fusion)
        img_fusion = Image.fromarray(np.uint8(img_fusion))
        img_fusion.save(save_path)


class test_gray(Test):
    def __init__(self):
        super().__init__()
        self.img_type = 'gray'

    def get_fusion(self, img1_path, img2_path, model,
                   save_path='none', save_name='none'):
        img1, img2 = self.load_imgs(img1_path, img2_path, device)

        fused_img = block_fusion(img1, img2, block_size=256)

        self.save_imgs(save_path, save_name, fused_img)
        return fused_img

def fun(test_path, model, save_path='./test_result/'):
    img_list = glob(test_path + '*')
    img_num = len(img_list) / 2
    suffix = img_list[0].split('.')[-1]
    img_name_list = list(
        set([img_list[i].split('\\')[-1].split('.')[0].strip(string.digits) for i in range(len(img_list))])) #for windows

    fusion_phase = test_gray()

    for i in range(int(img_num)):
        img1_path = test_path + img_name_list[0] + str(i) + '.' + suffix
        img2_path = test_path + img_name_list[1] + str(i) + '.' + suffix
        save_name = 'fusion_' + str(i) + '.' + suffix
        fusion_phase.get_fusion(img1_path, img2_path, model,
                                save_path=save_path, save_name=save_name)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser(description='model save and load')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0,1',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    args = parser.parse_args()
    device = 'cuda'

    model = TransNet().to(device)

    state_dict = torch.load('./best_model.pth', map_location='cuda:0')['model']

    if len(args.gpus) > 1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    test_path = './MEFB_L_gray/'

    model.eval()

    fun(test_path, model, save_path='./TransMEF_result1')
