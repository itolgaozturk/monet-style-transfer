# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
!pip install kaggle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import itertools
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from glob import glob

import copy
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#         break
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from google.colab import drive
drive.mount('/content/gdrive')

root_path = 'gdrive/My Drive/gan-dataset/'

for dirname, _, filenames in os.walk(root_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break

# Kaggle api does not download whole dataset, so not very useful
# from google.colab import files

# uploaded = files.upload()

# for fn in uploaded.keys():
#   print('User uploaded file "{name}" with length {length} bytes'.format(
#       name=fn, length=len(uploaded[fn])))
  
# # Then move kaggle.json into the folder where the API expects to find it.
# !mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json


# !!kaggle datasets download gan-getting-started

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

class Arguments():
    def __init__(self,dataroot=root_path, dataset=root_path+'monet_jpg/',  epochs=5, decay_epochs=1, batch_size=2, lr=0.0002, print_frequency=100, image_size=256, resultsFolder='gdrive/My Drive/gan-dataset/results/'):
        self.dataroot = dataroot
        self.dataset = dataset
        self.epochs = epochs
        self.decay_epochs = decay_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.print_frequency = print_frequency
        self.image_size = image_size
        self.resultsFolder = resultsFolder

args = Arguments()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)

class DecayLR:
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "start it before the prev training session!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (
                self.epochs - self.decay_epochs)

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_X = sorted(glob(os.path.join(args.dataroot, "monet_jpg/") + "/*.*")) # So X is defined as Monet images
        self.files_Y = sorted(glob(os.path.join(args.dataroot,  "photo_jpg/")+ "/*.*")) # Y is defined as photos
        print(self.files_X)
        
    def __getitem__(self, index):
        item_X = self.transform(Image.open(self.files_X[index % len(self.files_X)]))

        if self.unaligned:
            item_Y = self.transform(Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)]))
        else:
            item_Y = self.transform(Image.open(self.files_Y[index % len(self.files_Y)]))

        return {"X": item_X, "Y": item_Y}

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))

dataset = ImageDataset(root=args.dataroot,
                       transform=transforms.Compose([
                           transforms.Resize(int(args.image_size * 1.12), Image.BICUBIC),
                           transforms.RandomCrop(args.image_size),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                       unaligned=True)

len(dataset)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

try:
    os.makedirs(os.path.join(args.resultsFolder,  "X/"))
    os.makedirs(os.path.join(args.resultsFolder, "Y/"))
except OSError:
    pass

try:
    os.makedirs(root_path+'weights/')
except OSError:
    pass

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

# define models
netG_X2Y = Generator().to(device)
netG_Y2X = Generator().to(device)
netD_X = Discriminator().to(device)
netD_Y = Discriminator().to(device)

netG_X2Y.apply(weights_init)
netG_Y2X.apply(weights_init)
netD_X.apply(weights_init)
netD_Y.apply(weights_init)

def loadPrevModel(lastEpoch):
    lastEpoch = str(lastEpoch)
    netG_X2Y.load_state_dict(torch.load("gdrive/My Drive/gan-dataset/weights/netG_X2Y_epoch_"+lastEpoch+".pth"))
    netG_Y2X.load_state_dict(torch.load("gdrive/My Drive/gan-dataset/weights/netG_Y2X_epoch_"+lastEpoch+".pth"))
    netD_X.load_state_dict(torch.load("gdrive/My Drive/gan-dataset/weights/netD_X_epoch_"+lastEpoch+".pth"))
    netD_Y.load_state_dict(torch.load("gdrive/My Drive/gan-dataset/weights/netD_Y_epoch_"+lastEpoch+".pth"))
# use it only when you wanna resume previous training
loadPrevModel(4)

# define loss funcs
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

# Optimizers
optim_G = torch.optim.Adam(itertools.chain(netG_X2Y.parameters(), netG_Y2X.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
optim_D_X = torch.optim.Adam(netD_X.parameters(), lr=args.lr, betas=(0.5, 0.999))
optim_D_Y = torch.optim.Adam(netD_Y.parameters(), lr=args.lr, betas=(0.5, 0.999))

lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=lr_lambda)
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optim_D_X, lr_lambda=lr_lambda)
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optim_D_Y, lr_lambda=lr_lambda)

g_losses = []
d_losses = []

identity_losses = []
gan_losses = []
cycle_losses = []

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

generated_X_buffer = ReplayBuffer()
generated_Y_buffer = ReplayBuffer()

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        original_image_X = data["X"].to(device)
        original_image_Y = data["Y"].to(device)
        batch_size = original_image_X.size(0)

        # original data label is 1, generated data label is 0.
        original_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
        generated_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)


        # STEP 1: Update G network: Generators X2Y and Y2X

        # Set G_X and G_Y's gradients to zero
        optim_G.zero_grad()

        # Identity loss
        # G_Y2X(X) should equal X if original X is fed
        identity_image_X = netG_Y2X(original_image_X)
        identity_loss_X = identity_loss(identity_image_X, original_image_X) * 5.0
        # G_X2Y(Y) should equal Y if original Y is fed
        identity_image_Y = netG_X2Y(original_image_Y)
        identity_loss_Y = identity_loss(identity_image_Y, original_image_Y) * 5.0

        # GAN loss D_X(G_X(X))
        generated_image_X = netG_Y2X(original_image_Y)
        generated_output_X = netD_X(generated_image_X)
        loss_GAN_Y2X = adversarial_loss(generated_output_X, original_label)
        # GAN loss D_Y(G_Y(Y))
        generated_image_Y = netG_X2Y(original_image_X)
        generated_output_Y = netD_Y(generated_image_Y)
        loss_GAN_X2Y = adversarial_loss(generated_output_Y, original_label)

        # Cycle loss
        recovered_image_X = netG_Y2X(generated_image_Y)
        cycle_loss_XYX = cycle_loss(recovered_image_X, original_image_X) * 10.0

        recovered_image_Y = netG_X2Y(generated_image_X)
        cycle_loss_YXY = cycle_loss(recovered_image_Y, original_image_Y) * 10.0

        # Combined loss and calculate gradients
        error_G = identity_loss_X + identity_loss_Y + loss_GAN_X2Y + loss_GAN_Y2X + cycle_loss_XYX + cycle_loss_YXY

        # Calculate gradients for G_X and G_Y
        error_G.backward()
        # Update G_X and G_Y's weights
        optim_G.step()

        # STEP 2: Update D network: Discriminator X

        # Set D_X gradients to zero
        optim_D_X.zero_grad()

        # original X image loss
        original_output_X = netD_X(original_image_X)
        error_D_original_X = adversarial_loss(original_output_X, original_label)

        # generated X image loss
        generated_image_X = generated_X_buffer.push_and_pop(generated_image_X)
        generated_output_X = netD_X(generated_image_X.detach())
        error_D_generated_X = adversarial_loss(generated_output_X, generated_label)

        # Combined loss and calculate gradients
        error_D_X = (error_D_original_X + error_D_generated_X) / 2

        # Calculate gradients for D_X
        error_D_X.backward()
        # Update D_X weights
        optim_D_X.step()


        # STEP 3: Update D network: Discriminator Y

        # Set D_Y gradients to zero
        optim_D_Y.zero_grad()

        # original Y image loss
        original_output_Y = netD_Y(original_image_Y)
        error_D_original_Y = adversarial_loss(original_output_Y, original_label)

        # generated Y image loss
        generated_image_Y = generated_Y_buffer.push_and_pop(generated_image_Y)
        generated_output_Y = netD_Y(generated_image_Y.detach())
        error_D_generated_Y = adversarial_loss(generated_output_Y, generated_label)

        # Combined loss and calculate gradients
        error_D_Y = (error_D_original_Y + error_D_generated_Y) / 2

        # Calculate gradients for D_Y
        error_D_Y.backward()
        # Update D_Y weights
        optim_D_Y.step()

        progress_bar.set_description(
            f"[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(error_D_X + error_D_Y).item():.4f} "
            f"Loss_G: {error_G.item():.4f} "
            f"Loss_G_identity: {(identity_loss_X + identity_loss_Y).item():.4f} "
            f"loss_G_GAN: {(loss_GAN_X2Y + loss_GAN_Y2X).item():.4f} "
            f"loss_G_cycle: {(cycle_loss_XYX + cycle_loss_YXY).item():.4f}")

        if i % args.print_frequency == 0:
            vutils.save_image(original_image_X, f"{args.resultsFolder}X/original_samples.png", normalize=True)
            vutils.save_image(original_image_Y, f"{args.resultsFolder}Y/original_samples.png", normalize=True)

            generated_image_X = 0.5 * (netG_Y2X(original_image_Y).data + 1.0)
            generated_image_Y = 0.5 * (netG_X2Y(original_image_X).data + 1.0)

            vutils.save_image(generated_image_X.detach(),
                              f"{args.resultsFolder}X/generated_samples_epoch_{epoch}.png",
                              normalize=True)
            vutils.save_image(generated_image_Y.detach(),
                              f"{args.resultsFolder}Y/generated_samples_epoch_{epoch}.png",
                              normalize=True)

    # do check pointing
    torch.save(netG_X2Y.state_dict(), f"gdrive/My Drive/gan-dataset/weights/netG_X2Y_epoch_{epoch}.pth")
    torch.save(netG_Y2X.state_dict(), f"gdrive/My Drive/gan-dataset/weights/netG_Y2X_epoch_{epoch}.pth")
    torch.save(netD_X.state_dict(), f"gdrive/My Drive/gan-dataset/weights/netD_X_epoch_{epoch}.pth")
    torch.save(netD_Y.state_dict(), f"gdrive/My Drive/gan-dataset/weights/netD_Y_epoch_{epoch}.pth")

    # Update learning rates as decay manner
    lr_scheduler_G.step()
    lr_scheduler_D_X.step()
    lr_scheduler_D_Y.step()
