import os
import numpy as np
import torch
from torchvision import datasets, transforms


MEAN_MNIST, STD_MNIST = (0.1307,), (0.3081,)
MEAN_CIFAR10, STD_CIFAR10 = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
MEAN_IMAGENET, STD_IMAGENET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def normalize(env):
    if env == 'mnist':
        transform = transforms.Normalize(MEAN_MNIST, STD_MNIST)
    elif env == 'cifar10':
        transform = transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    else:
        raise NotImplementedError
    return transform


# preprocessing
def transform_cifar10():
    transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
    ])
    return transform


def transform_imagenet():
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)
            ])
    return transform


def transform_celebahq():
    transform = transforms.Compose([
                transforms.ToTensor()
            ])
    return transform


import os
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image
import json


class ImageNetKaggle(Dataset):
    def __init__(self, root='data/ILSVRC/', split='val', transform=transform_imagenet()):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


import pandas as pd


# get the attributes from celebahq subset
def make_table(root):
    filenames = sorted(os.listdir(f'{root}/celeba-256'))
    celebahq = [os.path.basename(f) for f in filenames]
    attr_gt = pd.read_csv(f'{root}/Anno/list_attr_celeba.txt',
                          skiprows=1, delim_whitespace=True, index_col=0)
    attr_celebahq = attr_gt.reindex(index=celebahq).replace(-1, 0)

    # get the train/test/val partitions
    partitions = {}
    with open(f'{root}/Anno/list_eval_partition.txt') as f:
        for line in f:
            filename, part = line.strip().split(' ')
            partitions[filename] = int(part)
    partitions_list = [partitions[fname] for fname in attr_celebahq.index]

    attr_celebahq['partition'] = partitions_list
    return attr_celebahq


class CelebAHQDataset(Dataset):
    def __init__(self, split='test', attribute='Eyeglasses', root='./data/celeba-hq/', transform=transform_celebahq()):
        self.transform = transform
        
        # make table
        attr_celebahq = make_table(root)

        # samples
        self.samples = []
        samples_dir = os.path.join(root, "celeba-256/")
        for entry in sorted(os.listdir(samples_dir)):
            self.samples.append(samples_dir + entry)

        # targets
        self.targets = attr_celebahq[attribute]

        # split
        split_to_int = dict(train=0, val=1, test=2)  # convert from train/val/test to partition numbers
        self.split_idx = np.where(attr_celebahq['partition'] == split_to_int[split])[0]
        self.samples = Subset(self.samples, self.split_idx)
        self.targets = self.targets.iloc[self.split_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


# data
def make_data(env, batch_size=50, test_batch_size=None, preprocess=False, n_start=0, n=None, seed=0):
    if test_batch_size is None:
        test_batch_size = batch_size
    if env == 'mnist':
        train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()
            ]))
        test_set = datasets.MNIST('./data',  train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
            ]))
    elif env == 'cifar10':
        train_set = datasets.CIFAR10('./data', train=True, download=True,
                                     transform=transform_cifar10() if preprocess else transforms.ToTensor())
        test_set = datasets.CIFAR10('./data',  train=False, transform=transforms.ToTensor())
    elif env == 'imagenet':
        train_set = None
        test_set = ImageNetKaggle('data/ILSVRC/', split='val', transform=transform_imagenet())
    elif env == 'celebahq':
        train_set = None
        test_set = CelebAHQDataset(split='test', attribute='Eyeglasses')
    else:
        raise NotImplementedError
    if n is not None:
        if train_set is not None:
            train_set = torch.utils.data.Subset(train_set, range(n_start, n_start + n))
        test_set = torch.utils.data.Subset(test_set, range(n_start, n_start + n))
        # if train_set is not None:
        #     np.random.seed(seed)
        #     subset = np.random.choice(np.arange(train_set.__len__()), size=n, replace=False)
        #     train_set = torch.utils.data.Subset(train_set, subset)
        # np.random.seed(seed)
        # subset = np.random.choice(np.arange(test_set.__len__()), size=n, replace=False)
        # test_set = Subset(test_set, subset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=1)
    return train_loader, test_loader
