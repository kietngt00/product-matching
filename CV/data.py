import numpy as np
import random
from collections import defaultdict 
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class CustomDataset(Dataset):
    def __init__(self, x, y, train = True, transform=None):
        self.x = x
        self.y = y

        self.transform = transform
        self.num_samples = len(self.x)
        self.train = train

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):

        path = self.x[idx]
        x = Image.open(path)
        y = self.y[idx]
                
        if self.transform is not None:
            x = self.transform(x)
        return x, y

class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = np.array([])
        for _, idx in enumerate(iter(self.sampler)):
            batch = idx
            yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) - 1) // self.batch_size + 1

class DynamicSampler(Sampler):
    def __init__(self, y, batch_size, label_per_batch):
        self.y = y
        self.batch_size = batch_size
        self.label_per_batch = label_per_batch

    def __iter__(self):
        dic = {}
        for name in set(self.y):
            dic[name] = set()
        for i in range(len(self.y)):
            dic[self.y[i]].add(i)
        num_batches = len(self.y)//self.batch_size
        while num_batches > 0:
            sampled = []
            while len(sampled) < self.batch_size:
                random_names = np.random.permutation(list(dic.keys()))
                for name in random_names:
                    selected_index = random.sample(dic[name], min(self.label_per_batch, len(dic[name]), self.batch_size - len(sampled)))
                    for idx in selected_index:
                        dic[name].remove(idx)
                    sampled.extend(selected_index)
                    if len(dic[name]) == 0:
                        del dic[name]
                    if len(sampled) == self.batch_size:
                        break
                    assert len(sampled) <= self.batch_size
            random.shuffle(sampled)
            yield sampled
            num_batches -=1

    def __len__(self):
        return len(self.y)