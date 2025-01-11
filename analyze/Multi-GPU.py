from datetime import datetime
import pytz
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import Net,run_check_net
from dataset import read_one_data, read_one_truth
from czii_helper import dotdict


B = 1
image_size = 640
mask_size  = 640
num_slice = 64 #184
num_class=6+1

batch = {
    'image': torch.from_numpy(np.random.uniform(0,1, (B,num_slice, image_size, image_size))).float(),
    'mask': torch.from_numpy(np.random.choice(num_class, (B, num_slice, mask_size, mask_size))).long(),
}

net = Net(pretrained=True, cfg=None).cuda()

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True):
        output = net(batch)
# ---
print('batch')
for k, v in batch.items():
    if k == 'D':
        print(f'{k:>32} : {v} ')
    else:
        print(f'{k:>32} : {v.shape} ')

print('output')
for k, v in output.items():
    if 'loss' not in k:
        print(f'{k:>32} : {v.shape} ')
print('loss')
for k, v in output.items():
    if 'loss' in k:
        print(f'{k:>32} : {v.item()} ')


