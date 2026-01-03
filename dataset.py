import os 
import numpy as np 
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
import torch

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'tiny_imagenet_dataloaders', 
            'cifar10_dataloaders_val', 'cifar100_dataloaders_val', 'tiny_imagenet_dataloaders_val',
            'gtsrb_dataloaders', 'gtsrb_dataloaders_val']


def cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10', dataset=False):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if dataset:
        print('return train dataset')
        train_dataset = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=128, data_dir='datasets/cifar100', dataset=False):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if dataset:
        print('return train dataset')
        train_dataset = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

class TinyImageNetValDataset(Dataset):
    """Tiny-ImageNet validation dataset with annotations file."""
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Read annotations file
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.images.append(parts[0])
                    self.labels.append(parts[1])
        
        # Create class to idx mapping from training folder
        train_path = os.path.join(os.path.dirname(os.path.dirname(root_dir)), 'train')
        train_dataset = ImageFolder(train_path)
        self.class_to_idx = train_dataset.class_to_idx
        
        # Convert class names to indices
        self.label_indices = [self.class_to_idx[label] for label in self.labels]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.label_indices[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def tiny_imagenet_dataloaders(batch_size=64, data_dir='datasets/tiny-imagenet-200', dataset=False, split_file=None):

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    val_images_path = os.path.join(val_path, 'images')
    val_annotations = os.path.join(val_path, 'val_annotations.txt')

    if not split_file:
        split_file = 'npy_files/tiny-imagenet-train-val.npy'
    split_permutation = list(np.load(split_file))

    train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = TinyImageNetValDataset(val_images_path, val_annotations, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if dataset:
        print('return train dataset')
        train_dataset = ImageFolder(train_path, transform=train_transform)
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

class TinyImageNetValDataset(Dataset):
    """Tiny-ImageNet validation dataset with annotations file."""
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Read annotations file
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.images.append(parts[0])
                    self.labels.append(parts[1])
        
        # Create class to idx mapping from training folder
        train_path = os.path.join(os.path.dirname(os.path.dirname(root_dir)), 'train')
        train_dataset = ImageFolder(train_path)
        self.class_to_idx = train_dataset.class_to_idx
        
        # Convert class names to indices
        self.label_indices = [self.class_to_idx[label] for label in self.labels]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.label_indices[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def cifar10_dataloaders_val(batch_size=128, data_dir='datasets/cifar10'):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader

def cifar100_dataloaders_val(batch_size=128, data_dir='datasets/cifar100'):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader

def tiny_imagenet_dataloaders_val(batch_size=64, data_dir='datasets/tiny-imagenet-200', split_file=None):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train')

    if not split_file:
        split_file = 'npy_files/tiny-imagenet-train-val.npy'
    split_permutation = list(np.load(split_file))

    train_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[:90000])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader


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


def gtsrb_dataloaders(batch_size=128, data_dir='data/GTSRB', dataset=False, split_file=None):

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'Train')
    test_path = os.path.join(data_dir, 'Test')
    test_csv = os.path.join(test_path, 'Test.csv')

    if not split_file:
        split_file = 'npy_files/gtsrb-train-val.npy'
    
    # Load or create split permutation
    if os.path.exists(split_file):
        split_permutation = list(np.load(split_file))
    else:
        # Create 90-10 split (35326 train, 3926 val from 39252 total)
        full_dataset = ImageFolder(train_path)
        total_size = len(full_dataset)
        split_permutation = np.random.permutation(total_size)
        os.makedirs('npy_files', exist_ok=True)
        np.save(split_file, split_permutation)
        print(f'Created new train-val split: {split_file}')
    
    train_size = int(len(split_permutation) * 0.9)
    
    train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:train_size])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[train_size:])
    
    # Use custom test dataset to handle flat structure
    test_set = GTSRBTestDataset(test_path, csv_file=test_csv, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if dataset:
        print('return train dataset')
        train_dataset = ImageFolder(train_path, transform=train_transform)
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader


def gtsrb_dataloaders_val(batch_size=128, data_dir='data/GTSRB', split_file=None):

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'Train')

    if not split_file:
        split_file = 'npy_files/gtsrb-train-val.npy'
    split_permutation = list(np.load(split_file))
    
    train_size = int(len(split_permutation) * 0.9)

    train_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[:train_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader



