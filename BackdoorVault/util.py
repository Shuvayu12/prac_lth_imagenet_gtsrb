import numpy as np
import os
import sys
import torch
import pandas as pd
from PIL import Image
from dfst import DFST
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Import external ResNet models from parent directory
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from models import ResNet18_CIFAR, ResNet18_GTSRB, ResNet18_TinyImageNet


EPSILON = 1e-7

_dataset_name = ['default', 'cifar10', 'gtsrb', 'imagenet']

_mean = {
    'default':  [0.5, 0.5, 0.5],
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'gtsrb':    [0.3337, 0.3064, 0.3171],
    'imagenet': [0.485, 0.456, 0.406],
}

_std = {
    'default':  [0.5, 0.5, 0.5],
    'cifar10':  [0.2023, 0.1994, 0.2010],
    'gtsrb':    [0.2672, 0.2564, 0.2629],
    'imagenet': [0.229, 0.224, 0.225],
}

_size = {
    'cifar10':  (32, 32),
    'gtsrb':    (32, 32),
    'imagenet': (224, 224),
    'tiny_imagenet': (64, 64),
}

_num = {
    'cifar10':  10,
    'gtsrb':    43,
    'tiny_imagenet': 200,
}


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, _dataset_name
        size = _size[size]
    return transforms.Resize(size)


def get_processing(dataset, augment=True, tensor=False, size=None):
    normalize, unnormalize = get_norm(dataset)

    transforms_list = []
    if size is not None:
        transforms_list.append(get_resize(size))
    if augment:
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        # For test data, resize to consistent size
        transforms_list.append(transforms.Resize(_size[dataset]))
    if not tensor:
        transforms_list.append(transforms.ToTensor())
    transforms_list.append(normalize)

    preprocess = transforms.Compose(transforms_list)
    deprocess  = transforms.Compose([unnormalize])
    return preprocess, deprocess


class GTSRBTestDataset(Dataset):
    """GTSRB test dataset with flat structure and CSV labels."""
    def __init__(self, root_dir, csv_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all ppm files
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.ppm')])
        
        # Load labels if CSV exists
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file, sep=';')
            self.labels = df['ClassId'].values
        else:
            # If no CSV, assume labels are not available (use -1 as placeholder)
            self.labels = [-1] * len(self.images)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    
def get_dataset(args, train=True, augment=True):
    transform, _ = get_processing(args.dataset, train & augment)
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(args.datadir, train, transform,
                                   download=False)
    elif args.dataset == 'gtsrb':
        if train:
            # Train: use ImageFolder with data/GTSRB/Train/ (has class subfolders)
            data_path = os.path.join(args.datadir, 'GTSRB', 'Train')
            dataset = datasets.ImageFolder(data_path, transform)
        else:
            # Test: use custom dataset for flat structure with CSV labels
            test_path = os.path.join(args.datadir, 'GTSRB', 'Test')
            csv_file = os.path.join(test_path, 'Test.csv')
            dataset = GTSRBTestDataset(test_path, csv_file, transform)
    elif args.dataset == 'tiny_imagenet':
        # Use ImageFolder with data/tiny-imagenet/train/ or data/tiny-imagenet/val/
        split_dir = 'train' if train else 'val'
        data_path = os.path.join(args.datadir, 'tiny-imagenet', split_dir)
        dataset = datasets.ImageFolder(data_path, transform)
    return dataset


def get_loader(args, train=True):
    dataset = get_dataset(args, train)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=4, shuffle=train)
    return dataloader


def get_model(network, dataset='cifar10'):
    """Get model based on network architecture and dataset.
    
    Uses external ResNet models from models/ directory.
    """
    if network == 'resnet18':
        if dataset == 'cifar10':
            model = ResNet18_CIFAR(num_classes=10)
        elif dataset == 'gtsrb':
            model = ResNet18_GTSRB(num_classes=43)
        elif dataset == 'tiny_imagenet':
            model = ResNet18_TinyImageNet(num_classes=200)
        else:
            model = ResNet18_CIFAR(num_classes=10)  # default to CIFAR
    else:
        raise ValueError(f"Network {network} not supported. Use 'resnet18'.")
    return model


def get_classes(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def get_backdoor(attack, shape, normalize=None, device=None, args=None):
    """Get DFST backdoor instance."""
    if args is not None:
        base_path = f'ckpt/{args.dataset}_{args.network}'
    else:
        base_path = ''
    
    if 'dfst' in attack:
        backdoor = DFST(normalize, device=device)
        genr_path = f'{base_path}_dfst_generator.pt'
        if os.path.exists(genr_path):
            backdoor.genr_a2b = torch.load(genr_path, map_location=device)
    else:
        backdoor = None
    return backdoor
