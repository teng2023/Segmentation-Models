from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
#import scipy.misc
import random
import os
import imageio

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

root_dir   = "fcn_Handmade/CityScapes/"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

num_class = 19
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 1024, 2048
train_h   = int(h/2)  # 512
train_w   = int(w/2)  # 1024
val_h     = h  # 1024
val_w     = w  # 2048


class CityScapesDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=num_class, crop=False, flip_rate=0.):
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class

        self.flip_rate = flip_rate
        self.crop      = crop
        if phase == 'train':
            self.crop = True
            self.flip_rate = 0.5
            self.new_h = train_h
            self.new_w = train_w

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]               #replace ix to ilo
        # img_name=os.path.join(img_name[0]+'gtFine'+img_name[1]+'gtFine_color'+img_name[2])  #added new line
        img        = imageio.imread(img_name, mode='RGB')    #replace scipy.misc to imageio
        label_name = self.data.iloc[idx, 1]
        label      = np.load(label_name)

        if self.crop:
            h, w, _ = img.shape
            top   = random.randint(0, h - self.new_h)
            left  = random.randint(0, w - self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        #label = torch.from_numpy(label.copy()).long()
        label=torch.from_numpy(label.copy()).int()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}    #target is a image of shape(n_class,h,w).
                                                        #label is a image of shape (h,w), and each pixel represents the label.
        return sample


def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


if __name__ == "__main__":
    train_data = CityScapesDataset(csv_file=train_file, phase='train')

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['Y'].size())
    
        # observe 4th batch
        if i == 3:
            plt.figure()
            show_batch(batch)
            plt.axis('off')
            #plt.ioff()
            plt.show()
            break
