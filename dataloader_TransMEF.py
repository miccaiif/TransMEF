# -*- coding: utf-8 -*-
# Citation:
# @article{qu2021transmef,
#   title={TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning},
#   author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
#   journal={arXiv preprint arXiv:2112.01030},
#   year={2021}
# }
from __future__ import print_function
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
from glob import glob
import os
import copy
from PIL import Image
import random
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
np.random.seed(2)


def local_pixel_shuffling(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x,
                                 block_noise_size_y))
        image_temp[noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def global_patch_shuffling(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)

    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)

        noise_x1 = random.randint(0, img_rows - block_noise_size_x)
        noise_y1 = random.randint(0, img_cols - block_noise_size_y)

        noise_x2 = random.randint(0, img_rows - block_noise_size_x)
        noise_y2 = random.randint(0, img_cols - block_noise_size_y)

        window1 = orig_image[noise_x1:noise_x1 + block_noise_size_x,
                  noise_y1:noise_y1 + block_noise_size_y]
        window2 = orig_image[noise_x2:noise_x2 + block_noise_size_x,
                  noise_y2:noise_y2 + block_noise_size_y]

        window_tmp = window1
        window1 = window2
        window2 = window_tmp

        image_temp[noise_x1:noise_x1 + block_noise_size_x,
        noise_y1:noise_y1 + block_noise_size_y] = window1
        image_temp[noise_x2:noise_x2 + block_noise_size_x,
        noise_y2:noise_y2 + block_noise_size_y] = window2

    local_shuffling_x = image_temp

    return local_shuffling_x


def brightness_aug(x, gamma):
    aug_brightness = iaa.Sequential(sometimes(iaa.GammaContrast(gamma=gamma)))
    aug_image = aug_brightness(images=x)
    return aug_image


def bright_transform(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y]
        window = brightness_aug(window, 3 * np.random.random_sample())

        image_temp[noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    bright_transform_x = image_temp

    return bright_transform_x


def fourier_broken(x, nb_rows, nb_cols):
    aug_a = iaa.GaussianBlur(sigma=0.5)
    aug_p = iaa.Jigsaw(nb_rows=nb_rows, nb_cols=nb_cols, max_steps=(1, 5))
    fre = np.fft.fft2(x)
    fre_a = np.abs(fre)
    fre_p = np.angle(fre)
    fre_a_normalize = fre_a / (np.max(fre_a) + 0.0001)
    fre_p_normalize = fre_p
    fre_a_trans = aug_a(image=fre_a_normalize)
    fre_p_trans = aug_p(image=fre_p_normalize)
    fre_a_trans = fre_a_trans * (np.max(fre_a) + 0.0001)
    fre_p_trans = fre_p_trans
    fre_recon = fre_a_trans * np.e ** (1j * (fre_p_trans))
    img_recon = np.abs(np.fft.ifft2(fre_recon))
    return img_recon


def fourier_transform(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y]
        window = fourier_broken(window, block_noise_size_x, block_noise_size_y)
        image_temp[noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    bright_transform_x = image_temp

    return bright_transform_x


class Fusionset(Data.Dataset):
    def __init__(self, io, args, root, transform=None, gray=True, partition='train', ssl_transformations=None):
        self.files = glob(os.path.join(root, '*.*'))
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transform
        self.ssl_transformations = ssl_transformations
        self.args = args

        if args.miniset == True:
            self.files = random.sample(self.files, int(args.minirate * len(self.files)))
        self.num_examples = len(self.files)

        if self.ssl_transformations == True:
            print('used ssl_transformations')
        else:
            print('not used ssl_transformations')

        if partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)
        io.cprint("number of " + partition + " examples in dataset" + ": " + str(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.gray:
            img = img.convert('L')
        img = np.array(img)

        if self.ssl_transformations == True:
            img_bright_orig = img.copy()
            img_bright_trans = bright_transform(img_bright_orig)
            img_bright_trans = self._tensor(img_bright_trans)

            img_fourier_orig = img.copy()
            img_fourier_trans = fourier_transform(img_fourier_orig)
            img_fourier_trans = self._tensor(img_fourier_trans)

            img_shuffling_orig = img.copy()
            img_shuffling_trans = global_patch_shuffling(img_shuffling_orig)
            img_shuffling_trans = self._tensor(img_shuffling_trans)
            img = self._tensor(img)
        else:
            img = self._tensor(img)
            img_bright_trans = img
            img_fourier_trans = img
            img_shuffling_trans = img

        return img, img_bright_trans, img_fourier_trans, img_shuffling_trans
