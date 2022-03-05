# -*- coding: utf-8 -*-
# Citation:
# @article{qu2021transmef,
#   title={TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning},
#   author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
#   journal={arXiv preprint arXiv:2112.01030},
#   year={2021}
# }
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import numpy as np
from Network_TransMEF import TransNet
from torchvision import transforms
from dataloader_TransMEF import Fusionset
from utils import mkdir
from ssim import SSIM, TV_Loss
import time
import argparse
import log
import copy
from tqdm import tqdm
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NWORKERS = 4

parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--exp_name', type=str, default='TransMEF_experiments', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--root', type=str, default='./coco', help='data path')
parser.add_argument('--save_path', type=str, default='./train_result_100all', help='model and pics save path')
parser.add_argument('--ssl_transformations', type=bool, default=True, help='use ssl_transformations or not')
parser.add_argument('--miniset', type=bool, default=False, help='to choose a mini dataset')
parser.add_argument('--minirate', type=float, default=0.2, help='to detemine the size of a mini dataset')
parser.add_argument('--seed', type=int, default=3, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--epoch', type=int, default=100, help='training epoch')
parser.add_argument('--batch_size', type=int, default=48, help='batchsize')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lamda_ssim', type=float, default=1, help='weight of the SSIM loss')
parser.add_argument('--lamda_tv', type=float, default=20, help='weight of the tv loss')
parser.add_argument('--lamda_mse', type=float, default=20, help='weight of the mse loss')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--summary_name', type=str, default='TransMEF_alldata_ssl_transformations_',
                    help='Name of the tensorboard summmary')

args = parser.parse_args()
writer = SummaryWriter(comment=args.summary_name)

# ==================
# init
# ==================
io = log.IOStream(args)
io.cprint(str(args))
toPIL = transforms.ToPILImage()
np.random.seed(1)  # to get the same images and leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# Read Data
# ==================
train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                     torchvision.transforms.RandomCrop(256),
                                                     torchvision.transforms.RandomHorizontalFlip()
                                                     ])

dataset = Fusionset(io, args, args.root, transform=train_augmentation, gray=True, partition='train',
                    ssl_transformations=args.ssl_transformations)

# Creating data indices for training and validation splits:
train_indices = dataset.train_ind
val_indices = dataset.val_ind

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)  # sampler will assign the whole data according to batchsize.
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                          sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                        sampler=valid_sampler)

torch.cuda.synchronize()
start = time.time()

# ==================
# Init Model
# ==================
model = TransNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = CosineAnnealingLR(optimizer, args.epoch)
MSE_fun = nn.MSELoss()
SSIM_fun = SSIM()
TV_fun = TV_Loss()

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)

# ==================
# Model Training
# ==================
loss_train = []
loss_val = []
mkdir(args.save_path)
print('============ Training Begins ===============')
model.train()

for epoch in tqdm(range(args.epoch)):
    all_loss_bright, mse_loss_bright, ssim_loss_bright, tv_loss_bright = 0., 0., 0., 0.
    all_loss_fourier, mse_loss_fourier, ssim_loss_fourier, tv_loss_fourier = 0., 0., 0., 0.
    all_loss_shuffling, mse_loss_shuffling, ssim_loss_shuffling, tv_loss_shuffling = 0., 0., 0., 0.

    total_task_loss_per_iter_refresh = 0.
    total_task_mse_loss_per_iter_refresh = 0.
    total_task_ssim_loss_per_iter_refresh = 0.
    total_task_tv_loss_per_iter_refresh = 0.

    total_task_loss_per_epoch_refresh = 0.
    total_task_mse_loss_per_epoch_refresh = 0.
    total_task_ssim_loss_per_epoch_refresh = 0.
    total_task_tv_loss_per_epoch_refresh = 0.

    total_bright_loss_per_epoch_refresh = 0.
    total_fourier_loss_per_epoch_refresh = 0.
    total_shuffling_loss_per_epoch_refresh = 0.

    for index, image in enumerate(train_loader):
        img_orig = image[0].to(device)
        img_bright_trans = image[1].to(device)  # shape:[B,1,256,256]
        img_fourier_trans = image[2].to(device)
        img_shuffling_trans = image[3].to(device)

        optimizer.zero_grad()

        img_recon_bright = model(img_bright_trans.float())
        img_recon_fourier = model(img_fourier_trans.float())
        img_recon_shuffling = model(img_shuffling_trans.float())

        image_bright = toPIL(img_recon_bright[0].squeeze(0).detach().cpu())
        image_fourier = toPIL(img_recon_fourier[0].squeeze(0).detach().cpu())
        image_shuffling = toPIL(img_recon_shuffling[0].squeeze(0).detach().cpu())

        if index % 100 == 0:
            image_bright.save(os.path.join(args.save_path, args.summary_name + '_bright_epoch' + str(epoch) + '_' + str(
                index) + '_coco_train.png'))
            image_fourier.save(
                os.path.join(args.save_path, args.summary_name + '_fourier_epoch' + str(epoch) + '_' + str(
                    index) + '_coco_train.png'))
            image_shuffling.save(
                os.path.join(args.save_path, args.summary_name + '_shuffling_epoch' + str(epoch) + '_' + str(
                    index) + '_coco_train.png'))

        mse_loss_bright = MSE_fun(img_orig, img_recon_bright)
        ssim_loss_bright = (1 - SSIM_fun(img_orig, img_recon_bright))
        tv_loss_bright = TV_fun(img_orig, img_recon_bright)
        all_loss_bright = args.lamda_tv * tv_loss_bright + args.lamda_ssim * ssim_loss_bright + args.lamda_mse * mse_loss_bright

        mse_loss_fourier = MSE_fun(img_orig, img_recon_fourier)
        ssim_loss_fourier = (1 - SSIM_fun(img_orig, img_recon_fourier))
        tv_loss_fourier = TV_fun(img_orig, img_recon_fourier)
        all_loss_fourier = args.lamda_tv * tv_loss_fourier + args.lamda_ssim * ssim_loss_fourier + args.lamda_mse * mse_loss_fourier

        mse_loss_shuffling = MSE_fun(img_orig, img_recon_shuffling)
        ssim_loss_shuffling = (1 - SSIM_fun(img_orig, img_recon_shuffling))
        tv_loss_shuffling = TV_fun(img_orig, img_recon_shuffling)
        all_loss_shuffling = args.lamda_tv * tv_loss_shuffling + args.lamda_ssim * ssim_loss_shuffling + args.lamda_mse * mse_loss_shuffling

        total_task_loss_per_iter_refresh = all_loss_bright + all_loss_fourier + all_loss_shuffling
        total_task_mse_loss_per_iter_refresh = mse_loss_bright + mse_loss_fourier + mse_loss_shuffling
        total_task_ssim_loss_per_iter_refresh = ssim_loss_bright + ssim_loss_fourier + ssim_loss_shuffling
        total_task_tv_loss_per_iter_refresh = tv_loss_bright + tv_loss_fourier + tv_loss_shuffling

        total_task_loss_per_iter_refresh.backward()
        optimizer.step()

        total_task_loss_per_epoch_refresh += total_task_loss_per_iter_refresh

        total_task_mse_loss_per_epoch_refresh += total_task_mse_loss_per_iter_refresh
        total_task_ssim_loss_per_epoch_refresh += total_task_ssim_loss_per_iter_refresh
        total_task_tv_loss_per_epoch_refresh += total_task_tv_loss_per_iter_refresh

        total_bright_loss_per_epoch_refresh += all_loss_bright
        total_fourier_loss_per_epoch_refresh += all_loss_fourier
        total_shuffling_loss_per_epoch_refresh += all_loss_shuffling

    print('Epoch:[%d/%d]-----Train------ LOSS:%.4f' % (
        epoch, args.epoch, total_task_loss_per_epoch_refresh / (len(train_loader))))
    writer.add_scalar('Train/total_task_loss', total_task_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_task_mse_loss', total_task_mse_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_task_ssim_loss', total_task_ssim_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_task_tv_loss', total_task_tv_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_bright_loss', total_bright_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_fourier_loss', total_fourier_loss_per_epoch_refresh / (len(train_loader)), epoch)
    writer.add_scalar('Train/total_shuffling_loss', total_shuffling_loss_per_epoch_refresh / (len(train_loader)), epoch)

    loss_train.append(total_task_loss_per_epoch_refresh / (len(train_loader)))
    scheduler.step()

    # ==================
    # Model Validation
    # ==================
    model.eval()
    with torch.no_grad():

        all_loss_bright, mse_loss_bright, ssim_loss_bright, tv_loss_bright = 0., 0., 0., 0.
        all_loss_fourier, mse_loss_fourier, ssim_loss_fourier, tv_loss_fourier = 0., 0., 0., 0.
        all_loss_shuffling, mse_loss_shuffling, ssim_loss_shuffling, tv_loss_shuffling = 0., 0., 0., 0.

        total_task_loss_per_iter_refresh = 0.
        total_task_mse_loss_per_iter_refresh = 0.
        total_task_ssim_loss_per_iter_refresh = 0.
        total_task_tv_loss_per_iter_refresh = 0.

        total_task_loss_per_epoch_refresh = 0.
        total_task_mse_loss_per_epoch_refresh = 0.
        total_task_ssim_loss_per_epoch_refresh = 0.
        total_task_tv_loss_per_epoch_refresh = 0.

        total_bright_loss_per_epoch_refresh = 0.
        total_fourier_loss_per_epoch_refresh = 0.
        total_shuffling_loss_per_epoch_refresh = 0.

        for index, image in enumerate(val_loader):
            img_orig = image[0].to(device)
            img_bright_trans = image[1].to(device)  # shape:[B,1,256,256]
            img_fourier_trans = image[2].to(device)
            img_shuffling_trans = image[3].to(device)

            img_recon_bright = model(img_bright_trans.float())
            img_recon_fourier = model(img_fourier_trans.float())
            img_recon_shuffling = model(img_shuffling_trans.float())

            mse_loss_bright = MSE_fun(img_orig, img_recon_bright)
            ssim_loss_bright = (1 - SSIM_fun(img_orig, img_recon_bright))
            tv_loss_bright = TV_fun(img_orig, img_recon_bright)
            all_loss_bright = args.lamda_tv * tv_loss_bright + args.lamda_ssim * ssim_loss_bright + args.lamda_mse * mse_loss_bright

            mse_loss_fourier = MSE_fun(img_orig, img_recon_fourier)
            ssim_loss_fourier = (1 - SSIM_fun(img_orig, img_recon_fourier))
            tv_loss_fourier = TV_fun(img_orig, img_recon_fourier)
            all_loss_fourier = args.lamda_tv * tv_loss_fourier + args.lamda_ssim * ssim_loss_fourier + args.lamda_mse * mse_loss_fourier

            mse_loss_shuffling = MSE_fun(img_orig, img_recon_shuffling)
            ssim_loss_shuffling = (1 - SSIM_fun(img_orig, img_recon_shuffling))
            tv_loss_shuffling = TV_fun(img_orig, img_recon_shuffling)
            all_loss_shuffling = args.lamda_tv * tv_loss_shuffling + args.lamda_ssim * ssim_loss_shuffling + args.lamda_mse * mse_loss_shuffling

            total_task_loss_per_iter_refresh = all_loss_bright + all_loss_fourier + all_loss_shuffling
            total_task_mse_loss_per_iter_refresh = mse_loss_bright + mse_loss_fourier + mse_loss_shuffling
            total_task_ssim_loss_per_iter_refresh = ssim_loss_bright + ssim_loss_fourier + ssim_loss_shuffling
            total_task_tv_loss_per_iter_refresh = tv_loss_bright + tv_loss_fourier + tv_loss_shuffling

            total_task_loss_per_epoch_refresh += total_task_loss_per_iter_refresh

            total_task_mse_loss_per_epoch_refresh += total_task_mse_loss_per_iter_refresh
            total_task_ssim_loss_per_epoch_refresh += total_task_ssim_loss_per_iter_refresh
            total_task_tv_loss_per_epoch_refresh += total_task_tv_loss_per_iter_refresh

            total_bright_loss_per_epoch_refresh += all_loss_bright
            total_fourier_loss_per_epoch_refresh += all_loss_fourier
            total_shuffling_loss_per_epoch_refresh += all_loss_shuffling

        print('Epoch:[%d/%d]-----Val------ LOSS:%.4f' % (
            epoch, args.epoch, total_task_loss_per_epoch_refresh / (len(val_loader))))
        writer.add_scalar('Val/total_task_loss', total_task_loss_per_epoch_refresh / (len(val_loader)), epoch)
        writer.add_scalar('Val/total_task_mse_loss', total_task_mse_loss_per_epoch_refresh / (len(val_loader)),
                          epoch)
        writer.add_scalar('Val/total_task_ssim_loss', total_task_ssim_loss_per_epoch_refresh / (len(val_loader)),
                          epoch)
        writer.add_scalar('Val/total_task_tv_loss', total_task_tv_loss_per_epoch_refresh / (len(val_loader)), epoch)
        writer.add_scalar('Val/total_bright_loss', total_bright_loss_per_epoch_refresh / (len(val_loader)), epoch)
        writer.add_scalar('Val/total_fourier_loss', total_fourier_loss_per_epoch_refresh / (len(val_loader)), epoch)
        writer.add_scalar('Val/total_shuffling_loss', total_shuffling_loss_per_epoch_refresh / (len(val_loader)),
                          epoch)

        loss_val.append(total_task_loss_per_epoch_refresh / (len(val_loader)))

    # ==================
    # Model Saving
    # ==================
    # save model every epoch
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
    }
    torch.save(state, os.path.join(args.save_path,
                                   args.summary_name + 'epoch_' + str(epoch) + '_' + str(loss_val[epoch]) + '.pth'))
torch.cuda.synchronize()
end = time.time()

# save best model
minloss_index = loss_val.index(min(loss_val))
print("The min loss in validation is obtained in %d epoch" % (minloss_index))
print("The training process has finished! Take a break! ")
