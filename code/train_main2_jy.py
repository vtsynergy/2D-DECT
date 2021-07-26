#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/31/2020 1:43 PM
# @Author : Zhicheng Zhang
# @E-mail : zhicheng0623@gmail.com
# @Site :
# @File : train_main.py
# @Software: PyCharm
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
from skimage import io, transform
from skimage import img_as_float
import os
from os import path
from PIL import Image
from csv import reader
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re

INPUT_CHANNEL_SIZE = 1


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def read_correct_image(path):
    offset = 0
    ct_org = None
    with Image.open(path) as img:
        ct_org = np.float32(np.array(img))
        if 270 in img.tag.keys():
            for item in img.tag[270][0].split("\n"):
                if "c0=" in item:
                    loi = item.strip()
                    offset = re.findall(r"[-+]?\d*\.\d+|\d+", loi)
                    offset = (float(offset[1]))
    ct_org = ct_org + offset
    neg_val_index = ct_org < (-1024)
    ct_org[neg_val_index] = -1024
    return ct_org


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.001:
            min_val = -0.1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (batch, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights


    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    if(torch.isnan(output)):
        print(pow1)
        print(pow2)
        print(ssims)
        print(mcs)
        exit()


    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize="simple")
        #return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class denseblock(nn.Module):
    def __init__(self,nb_filter=16,filter_wh = 5):
        super(denseblock, self).__init__()
        self.input = None                           ######CHANGE
        self.nb_filter = nb_filter
        self.nb_filter_wh = filter_wh
        ##################CHANGE###############
        self.conv1_0 = nn.Conv2d(in_channels=nb_filter,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_0 = nn.Conv2d(in_channels=self.conv1_0.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_1 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_1.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_2 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv1_2.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_3 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_3 = nn.Conv2d(in_channels=self.conv1_3.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1 = [self.conv1_0, self.conv1_1, self.conv1_2, self.conv1_3]
        self.conv2 = [self.conv2_0, self.conv2_1, self.conv2_2, self.conv2_3]

        self.batch_norm1_0 = nn.BatchNorm2d(nb_filter)
        self.batch_norm2_0 = nn.BatchNorm2d(self.conv1_0.out_channels)
        self.batch_norm1_1 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels)
        self.batch_norm2_1 = nn.BatchNorm2d(self.conv1_1.out_channels)
        self.batch_norm1_2 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels)
        self.batch_norm2_2 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.batch_norm1_3 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels)
        self.batch_norm2_3 = nn.BatchNorm2d(self.conv1_3.out_channels)

        self.batch_norm1 = [self.batch_norm1_0, self.batch_norm1_1, self.batch_norm1_2, self.batch_norm1_3]
        self.batch_norm2 = [self.batch_norm2_0, self.batch_norm2_1, self.batch_norm2_2, self.batch_norm2_3]


    #def Forward(self, inputs):
    def forward(self, inputs):                      ######CHANGE
        #x = self.input
        x = inputs
        for i in range(4):
            #conv = nn.BatchNorm2d(x.size()[1])(x)
            conv = self.batch_norm1[i](x)
            #if(self.conv1[i].weight.grad != None ):
            #    print("weight_grad_" + str(i) + "_1", self.conv1[i].weight.grad.max())
            conv = self.conv1[i](conv)      ######CHANGE
            conv = F.leaky_relu(conv)


            #conv = nn.BatchNorm2d(conv.size()[1])(conv)
            conv = self.batch_norm2[i](conv)
            #if(self.conv2[i].weight.grad != None ):
            #    print("weight_grad_" + str(i) + "_2", self.conv2[i].weight.grad.max())
            conv = self.conv2[i](conv)      ######CHANGE
            conv = F.leaky_relu(conv)
            x = torch.cat((x, conv),dim=1)

        return x

class DD_net(nn.Module):
    def __init__(self):
        super(DD_net, self).__init__()
        self.input = None                       #######CHANGE
        self.nb_filter = 16

        ##################CHANGE###############
        self.conv1 = nn.Conv2d(in_channels=INPUT_CHANNEL_SIZE, out_channels=self.nb_filter, kernel_size=(7, 7), padding = (3,3))
        self.dnet1 = denseblock(self.nb_filter,filter_wh=5)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels*5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet2 = denseblock(self.nb_filter,filter_wh=5)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels*5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet3 = denseblock(self.nb_filter, filter_wh=5)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels*5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet4 = denseblock(self.nb_filter, filter_wh=5)

        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels*5, out_channels=self.nb_filter, kernel_size=(1, 1))

        self.convT1 = nn.ConvTranspose2d(in_channels=self.conv4.out_channels + self.conv4.out_channels,out_channels=2*self.nb_filter,kernel_size=5, padding=(2, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=self.convT1.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT3 = nn.ConvTranspose2d(in_channels=self.convT2.out_channels + self.conv3.out_channels,out_channels=2*self.nb_filter,kernel_size=5, padding=(2, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=self.convT3.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT5 = nn.ConvTranspose2d(in_channels=self.convT4.out_channels + self.conv2.out_channels,out_channels=2*self.nb_filter,kernel_size=5, padding=(2, 2))
        self.convT6 = nn.ConvTranspose2d(in_channels=self.convT5.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT7 = nn.ConvTranspose2d(in_channels=self.convT6.out_channels + self.conv1.out_channels,out_channels=2*self.nb_filter,kernel_size=5, padding=(2, 2))
        self.convT8 = nn.ConvTranspose2d(in_channels=self.convT7.out_channels,out_channels=1 ,kernel_size=1)
        self.batch1 = nn.BatchNorm2d(1)
        self.max1 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.batch2 = nn.BatchNorm2d(self.nb_filter*5)
        self.max2 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.batch3 = nn.BatchNorm2d(self.nb_filter*5)
        self.max3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
        self.batch4 = nn.BatchNorm2d(self.nb_filter*5)
        self.max4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
        self.batch5 = nn.BatchNorm2d(self.nb_filter*5)

        self.batch6 = nn.BatchNorm2d(self.conv5.out_channels+self.conv4.out_channels)
        self.batch7 = nn.BatchNorm2d(self.convT1.out_channels)
        self.batch8 = nn.BatchNorm2d(self.convT2.out_channels+self.conv3.out_channels)
        self.batch9 = nn.BatchNorm2d(self.convT3.out_channels)
        self.batch10 = nn.BatchNorm2d(self.convT4.out_channels+self.conv2.out_channels)
        self.batch11 = nn.BatchNorm2d(self.convT5.out_channels)
        self.batch12 = nn.BatchNorm2d(self.convT6.out_channels+self.conv1.out_channels)
        self.batch13 = nn.BatchNorm2d(self.convT7.out_channels)


    #def Forward(self, inputs):
    def forward(self, inputs):

        self.input = inputs
        #conv = nn.BatchNorm2d(self.input)
        conv = self.batch1(self.input)        #######CHANGE
        #conv = nn.Conv2d(in_channels=conv.get_shape().as_list()[1], out_channels=self.nb_filter, kernel_size=(7, 7))(conv)
        conv = self.conv1(conv)         #####CHANGE
        c0 = F.leaky_relu(conv)

        p0 = self.max1(c0)
        D1 = self.dnet1(p0)

        #######################################################################################
        conv = self.batch2(D1)
        conv = self.conv2(conv)
        c1 = F.leaky_relu(conv)

        p1 = self.max2(c1)
        D2 = self.dnet2(p1)
        #######################################################################################

        conv = self.batch3(D2)
        conv = self.conv3(conv)
        c2 = F.leaky_relu(conv)

        p2 = self.max3(c2)
        D3 = self.dnet3(p2)
        #######################################################################################

        conv = self.batch4(D3)
        conv = self.conv4(conv)
        c3 = F.leaky_relu(conv)

        #p3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)(c3)
        p3 = self.max4(c3)        ######CHANGE
        D4 = self.dnet4(p3)

        conv = self.batch5(D4)
        conv = self.conv5(conv)
        c4 = F.leaky_relu(conv)

        x = torch.cat((nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)(c4), c3),dim=1)
        dc4 = F.leaky_relu(self.convT1(self.batch6(x)))         ######size() CHANGE
        dc4_1 = F.leaky_relu(self.convT2(self.batch7(dc4)))

        x = torch.cat((nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)(dc4_1), c2),dim=1)
        dc5 = F.leaky_relu(self.convT3(self.batch8(x)))
        dc5_1 = F.leaky_relu(self.convT4(self.batch9(dc5)))

        x = torch.cat((nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)(dc5_1), c1),dim=1)
        dc6 = F.leaky_relu(self.convT5(self.batch10(x)))
        dc6_1 = F.leaky_relu(self.convT6(self.batch11(dc6)))

        x = torch.cat((nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)(dc6_1), c0),dim=1)
        dc7 = F.leaky_relu(self.convT7(self.batch12(x)))
        dc7_1 = F.leaky_relu(self.convT8(self.batch13(dc7)))

        output = dc7_1

        return  output

def gen_visualization_files(outputs, targets, inputs, file_names, val_test, maxs, mins):
    mapped_root = "./visualize/" + val_test + "/mapped/"
    diff_target_out_root = "./visualize/" + val_test + "/diff_target_out/"
    diff_target_in_root = "./visualize/" + val_test + "/diff_target_in/"
    ssim_root = "./visualize/" + val_test + "/ssim/"
    out_root = "./visualize/" + val_test + "/"
    in_img_root = "./visualize/" + val_test + "/input/"
    out_img_root = "./visualize/" + val_test + "/target/"

    if not os.path.exists("./visualize"):
        os.makedirs("./visualize")
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    if not os.path.exists(mapped_root):
        os.makedirs(mapped_root)
    if not os.path.exists(diff_target_in_root):
        os.makedirs(diff_target_in_root)
    if not os.path.exists(diff_target_out_root):
        os.makedirs(diff_target_out_root)
    if not os.path.exists(in_img_root):
        os.makedirs(in_img_root)
    if not os.path.exists(out_img_root):
        os.makedirs(out_img_root)

    MSE_loss_out_target = []
    MSE_loss_in_target = []
    MSSSIM_loss_out_target = []
    MSSSIM_loss_in_target = []


    outputs_size = list(outputs.size())
    #num_img = outputs_size[0]
    (num_img, channel, height, width) = outputs.size()
    for i in range(num_img):
        #output_img = outputs[i, 0, :, :].cpu().detach().numpy()
        output_img = outputs[i, 0, :, :].cpu().detach().numpy()
        target_img = targets[i, 0, :, :].cpu().numpy()
        input_img = inputs[i, 0, :, :].cpu().numpy()

        #jy
        # output_img = (output_img- (output_img.min().item()))/(output_img.max().item() - output_img.min().item())
        # output_img_mapped = (output_img * 1500) - 1000
        # target_img_mapped = (target_img * 1500) - 1000
        # input_img_mapped = (input_img * 1500) - 1000
        output_img_mapped = (output_img * (maxs[i].item() - mins[i].item())) + mins[i].item()
        target_img_mapped = (target_img * (maxs[i].item() - mins[i].item())) + mins[i].item()
        input_img_mapped = (input_img * (maxs[i].item() - mins[i].item())) + mins[i].item()

        # target_img = targets[i, 0, :, :].cpu().numpy()
        # input_img = inputs[i, 0, :, :].cpu().numpy()

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(target_img_mapped)
        #jy
        # im.save(out_img_root + file_name)

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(input_img_mapped)
        #jy
        # im.save(in_img_root + file_name)
        im.save(folder_ori_HU+'/'+file_name)

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(output_img_mapped)
        #jy
        # im.save(mapped_root + file_name)
        im.save(folder_enh_HU+'/'+file_name)

        difference_target_out = (target_img - output_img)
        difference_target_out = np.absolute(difference_target_out)
        fig = plt.figure()
        plt.imshow(difference_target_out)
        plt.colorbar()
        plt.clim(0,0.6)
        plt.axis('off')
        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        #jy
        # fig.savefig(diff_target_out_root + file_name)
        plt.clf()
        plt.close()

        difference_target_in = (target_img - input_img)
        difference_target_in = np.absolute(difference_target_in)
        fig = plt.figure()
        plt.imshow(difference_target_in)
        plt.colorbar()
        plt.clim(0,0.6)
        plt.axis('off')
        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        #jy
        # fig.savefig(diff_target_in_root + file_name)
        plt.clf()
        plt.close()

        output_img = torch.reshape(outputs[i, 0, :, :], (1, 1, height, width))
        target_img = torch.reshape(targets[i, 0, :, :], (1, 1, height, width))
        input_img = torch.reshape(inputs[i, 0, :, :], (1, 1, height, width))

        MSE_loss_out_target.append(nn.MSELoss()(output_img, target_img))
        MSE_loss_in_target.append(nn.MSELoss()(input_img, target_img))
        MSSSIM_loss_out_target.append(1 - MSSSIM()(output_img, target_img))
        MSSSIM_loss_in_target.append(1 - MSSSIM()(input_img, target_img))

    with open(out_root + "msssim_loss_target_out", 'a') as f:
        for item in MSSSIM_loss_out_target:
            f.write("%f\n" % item)

    with open(out_root + "msssim_loss_target_in", 'a') as f:
        for item in MSSSIM_loss_in_target:
            f.write("%f\n" % item)

    with open(out_root + "mse_loss_target_out", 'a') as f:
        for item in MSE_loss_out_target:
            f.write("%f\n" % item)

    with open(out_root + "mse_loss_target_in", 'a') as f:
        for item in MSE_loss_in_target:
            f.write("%f\n" % item)

class CTDataset(Dataset):
    def __init__(self, root_dir_h, root_dir_l, transform=None):
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
        self.img_list_l = os.listdir(self.data_root_l)
        self.img_list_h = os.listdir(self.data_root_h)
        self.transform = transform
    def __len__(self):
        return len(self.img_list_l)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs_np = None
        targets_np = None
        rmin = 0
        rmax = 1
        image_target = read_correct_image(self.data_root_h + self.img_list_l[idx])
        image_input = read_correct_image(self.data_root_h + self.img_list_h[idx])

        input_file = self.img_list_l[idx]
        assert(image_input.shape[0] == 512 and image_input.shape[1] == 512)
        assert(image_target.shape[0] == 512 and image_target.shape[1] == 512)
        cmax1 = np.amax(image_target)
        cmin1 = np.amin(image_target)
        image_target = rmin + ((image_target - cmin1)/(cmax1 - cmin1)*(rmax - rmin))
        assert((np.amin(image_target)>=0) and (np.amax(image_target)<=1))
        cmax2 = np.amax(image_input)
        cmin2 = np.amin(image_input)
        image_input = rmin + ((image_input - cmin2)/(cmax2 - cmin2)*(rmax - rmin))
        assert((np.amin(image_input)>=0) and (np.amax(image_input)<=1))
        mins = ((cmin1+cmin2)/2)
        maxs = ((cmax1+cmax2)/2)
        image_target = image_target.reshape((1, 512, 512))
        image_input = image_input.reshape((1, 512, 512))
        inputs_np = image_input
        targets_np = image_target

        inputs = torch.from_numpy(inputs_np)
        targets = torch.from_numpy(targets_np)
        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)

        sample = {'vol': input_file,
                  'HQ': targets,
                  'LQ': inputs,
                  'max': maxs,
                  'min': mins}
        return sample

#jy
# if __name__ == '__main__':
def __main__(add):

    ####################DATA DIRECTORY###################
    #jy
    global root
    # root = "/groups/synergy_lab/garvit217/transfer/mayo_clinic_data_tif/L067/full_1mm"
    root = add

    trainset = CTDataset(root_dir_h=root, root_dir_l=root)
    testset = CTDataset(root_dir_h=root, root_dir_l=root)
    valset = CTDataset(root_dir_h=root, root_dir_l=root)


    path_in_lists = root.split('/')
    folder_name = path_in_lists[len(path_in_lists)-2]
    choose_source = ''
    for i in range(len(path_in_lists)-3):
        choose_source = choose_source + '/' + path_in_lists[i]
    global folder_enh_HU
    global folder_ori_HU
    folder_enh_HU = choose_source + '/HU_enhanced/' + folder_name
    folder_ori_HU = choose_source + '/HU_original/' + folder_name
    if not os.path.exists(folder_enh_HU):
        os.mkdir(folder_enh_HU)
    if not os.path.exists(folder_ori_HU):
        os.mkdir(folder_ori_HU)

    # data_root_l = '/home/jingyuan/Documents/COVID-19_project/COVID-CT-master_NEW/Images-processed/origional_data/Europ_fullmask/image/sub-S04092_ses-E08271_run-3_bp-chest_ct_nii0357/'
    # data_root_h = '/home/jingyuan/Documents/COVID-19_project/COVID-CT-master_NEW/Images-processed/origional_data/Europ_fullmask/image/sub-S04092_ses-E08271_run-3_bp-chest_ct_nii0357/'
    #####################################################
    #data_root_h = "/groups/synergy_lab/garvit217/covid_sim_data/detect_angle/res2/covid_orig/"

    batch = 1
    epochs = 50

    train_loader = DataLoader(trainset, batch_size=batch, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch, drop_last=False, shuffle=False)
    val_loader = DataLoader(valset, batch_size=batch, drop_last=False, shuffle=False)

    mycwd = os.path.split(os.path.realpath(__file__))[0]
    model_file = mycwd[:-5]+"/code/weights_" + str(epochs) + "_" + str(batch) + ".pt"
    transfer_file = "transfer_weight.pt"


    # if not os.path.exists("./loss"):
    #     os.makedirs("./loss")
    #jy
    # if not os.path.exists("./reconstructed_images/val"):
    #     os.makedirs("./reconstructed_images/val")
    # if not os.path.exists("./reconstructed_images/test"):
    #     os.makedirs("./reconstructed_images/test")
    # if not os.path.exists("./reconstructed_images"):
    #     os.makedirs("./reconstructed_images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device: ", device)

    model = DD_net()
    param = model.parameters()
    print("Parameters length: ", len(list(param)))
    #for name, param in model.named_parameters():
    #    print(name)

    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)     #######ADAM CHANGE
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    learn_rate = 0.0001;
    epsilon = 1e-8

    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, eps=epsilon)      #######ADAM CHANGE
    optimizer1 = torch.optim.Adam(model.dnet1.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    optimizer2 = torch.optim.Adam(model.dnet2.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    optimizer3 = torch.optim.Adam(model.dnet3.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    optimizer4 = torch.optim.Adam(model.dnet4.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE

    decayRate = 0.8
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer1, gamma=decayRate)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer2, gamma=decayRate)
    scheduler3 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer3, gamma=decayRate)
    scheduler4 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer4, gamma=decayRate)

    train_MSE_loss = []
    train_MSSSIM_loss = []
    train_total_loss = []

    val_MSE_loss = []
    val_MSSSIM_loss = []
    val_total_loss = []

    test_MSE_loss = []
    test_MSSSIM_loss = []
    test_total_loss = []

    if (path.exists(transfer_file)):
        print("Loading model parameters for tranfer learning")
        model.load_state_dict(torch.load(transfer_file, map_location=device))

    model.to(device)




    if (not(path.exists(model_file))):

        for k in range(epochs):
            print("Epoch: ", k)
            for batch_index, batch_samples in enumerate(train_loader):
                file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], batch_samples['max'], batch_samples['min']
                inputs = LQ_img.to(device)
                targets = HQ_img.to(device)
                outputs = model(inputs)
                MSE_loss = nn.MSELoss()(outputs , targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                #loss = nn.MSELoss()(outputs , targets_train) + 0.1*(1-MSSSIM()(outputs,targets_train))
                loss = MSE_loss + 0.1*(MSSSIM_loss)
                #loss = MSE_loss
                #print("MSE_loss", MSE_loss.item())
                #print("MSSSIM_loss", MSSSIM_loss.item())
                #print("Total_loss", loss.item())

                train_MSE_loss.append(MSE_loss.item())
                train_MSSSIM_loss.append(MSSSIM_loss.item())
                train_total_loss.append(loss.item())

                model.zero_grad()
                model.dnet1.zero_grad()
                model.dnet2.zero_grad()
                model.dnet3.zero_grad()
                model.dnet4.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()

            scheduler.step()
            scheduler1.step()
            scheduler2.step()
            scheduler3.step()
            scheduler4.step()


            for batch_index, batch_samples in enumerate(val_loader):
                file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], batch_samples['max'], batch_samples['min']
                inputs = LQ_img.to(device)
                targets = HQ_img.to(device)
                outputs = model(inputs)
                MSE_loss = nn.MSELoss()(outputs , targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                #loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                loss = MSE_loss + 0.1*(MSSSIM_loss)
                #loss = MSE_loss
                #print("MSE_loss", MSE_loss.item())
                #print("MSSSIM_loss", MSSSIM_loss.item())
                #print("Total_loss", loss.item())
                #print("====================================")
                val_MSE_loss.append(MSE_loss.item())
                val_MSSSIM_loss.append(MSSSIM_loss.item())
                val_total_loss.append(loss.item())


                if(k==epochs-1):
                    outputs_np = outputs.cpu().detach().numpy()
                    (batch_size, channel, height, width) = outputs.size()
                    for m in range(batch_size):
                        file_name1 = file_name[m]
                        file_name1 = file_name1.replace(".IMA", ".tif")
                        im = Image.fromarray(outputs_np[m, 0, :, :])
                        #jy
                        # im.save('reconstructed_images/val/' + file_name1)

                    #gen_visualization_files(outputs, targets, inputs, val_files[l_map:l_map+batch], "val")
                    gen_visualization_files(outputs, targets, inputs, file_name, "val", maxs, mins)

        print("train end")

        print("Saving model parameters")
        torch.save(model.state_dict(), model_file)

        with open('loss/train_MSE_loss', 'w') as f:
            for item in train_MSE_loss:
                f.write("%f " % item)

        with open('loss/train_MSSSIM_loss', 'w') as f:
            for item in train_MSSSIM_loss:
                f.write("%f " % item)

        with open('loss/train_total_loss', 'w') as f:
            for item in train_total_loss:
                f.write("%f " % item)

        with open('loss/val_MSE_loss', 'w') as f:
            for item in val_MSE_loss:
                f.write("%f " % item)

        with open('loss/val_MSSSIM_loss', 'w') as f:
            for item in val_MSSSIM_loss:
                f.write("%f " % item)

        with open('loss/val_total_loss', 'w') as f:
            for item in val_total_loss:
                f.write("%f " % item)

    else:
        print("Loading saved model parameters")
        model.load_state_dict(torch.load(model_file, map_location=device))

    for batch_index, batch_samples in enumerate(test_loader):
        file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], batch_samples['max'], batch_samples['min']
        inputs = LQ_img.to(device)
        targets = HQ_img.to(device)
        outputs = model(inputs)
        MSE_loss = nn.MSELoss()(outputs , targets)
        MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
        #loss = nn.MSELoss()(outputs , targets_test) + 0.1*(1-MSSSIM()(outputs,targets_test))
        loss = MSE_loss + 0.1*(MSSSIM_loss)
        #loss = MSE_loss
        #print("MSE_loss", MSE_loss.item())
        #print("MSSSIM_loss", MSSSIM_loss.item())
        #print("Total_loss", loss.item())
        #print("====================================")
        test_MSE_loss.append(MSE_loss.item())
        test_MSSSIM_loss.append(MSSSIM_loss.item())
        test_total_loss.append(loss.item())

        outputs_np = outputs.cpu().detach().numpy()
        (batch_size, channel, height, width) = outputs.size()
        for m in range(batch_size):
            file_name1 = file_name[m]
            file_name1 = file_name1.replace(".IMA", ".tif")
            im = Image.fromarray(outputs_np[m, 0, :, :])
            #jy
            # im.save('reconstructed_images/val/' + file_name1)

        #outputs.cpu()
        #targets_test[l_map:l_map+batch, :, :, :].cpu()
        #inputs_test[l_map:l_map+batch, :, :, :].cpu()

        #gen_visualization_files(outputs, targets, inputs, test_files[l_map:l_map+batch], "test" )
        gen_visualization_files(outputs, targets, inputs, file_name, "test", maxs, mins)

    
    print("Testing end")
    print("Enhanced images can be found in " + folder_enh_HU)


    # with open('loss/test_MSE_loss', 'w') as f:
    #     for item in test_MSE_loss:
    #         f.write("%f " % item)
    #
    # with open('loss/test_MSSSIM_loss', 'w') as f:
    #     for item in test_MSSSIM_loss:
    #         f.write("%f " % item)
    #
    # with open('loss/test_total_loss', 'w') as f:
    #     for item in test_total_loss:
    #         f.write("%f " % item)
