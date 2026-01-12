"""
Dataset module for DFST backdoor attack.
Contains PoisonDataset and ImageDataset for DFST style transfer attack.
"""
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class PoisonDataset(Dataset):
    """Dataset wrapper for DFST poisoned data generation."""
    
    def __init__(self, dataset, data_rate=1.0, attack='dfst', target=0, 
                 poison_rate=1.0, processing=(None, None), transform=None, 
                 backdoor=None):
        """
        Args:
            dataset: Base dataset to poison
            data_rate: Fraction of dataset to use
            attack: Attack type (dfst)
            target: Target label for poisoned samples
            poison_rate: Fraction of samples to poison
            processing: Tuple of (normalize, unnormalize) transforms
            transform: Additional transforms to apply
            backdoor: DFST backdoor instance
        """
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.attack = attack
        self.target = target
        self.transform = transform
        self.processing = processing
        self.backdoor = backdoor

        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_poison = int(L * poison_rate)
        self.n_normal = self.n_data - self.n_poison

        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        # Get labels - handle different dataset types
        if hasattr(self.dataset, 'targets'):
            basic_labels = np.array(self.dataset.targets)[self.basic_index]
        elif hasattr(self.dataset, 'labels'):
            basic_labels = np.array(self.dataset.labels)[self.basic_index]
        else:
            # For ImageFolder-style datasets
            basic_labels = np.array([self.dataset[i][1] for i in self.basic_index])
        
        self.uni_index = {}
        for i in np.unique(basic_labels):
            self.uni_index[i] = np.where(i == np.array(basic_labels))[0].tolist()

    def __getitem__(self, index):
        i = np.random.randint(0, self.n_data)
        img, lbl = self.dataset[i]
        
        if index < self.n_poison:
            # Poison this sample - avoid using samples with target label
            while lbl == self.target:
                i = np.random.randint(0, self.n_data)
                img, lbl = self.dataset[i]
            lbl = self.target
            img = self.inject_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl

    def __len__(self):
        return self.n_normal + self.n_poison

    def inject_trigger(self, img):
        """Inject DFST style transfer trigger into image."""
        img = img.unsqueeze(0)
        if 'dfst' in self.attack:
            img = self.backdoor.inject(img)[0]
        else:
            img = img[0]
        return img


class ImageDataset(Dataset):
    """Dataset for loading images from a directory (for style images in DFST)."""
    
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir: Directory containing images
            transform: Transform to apply to images
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        image = read_image(img_path) / 255.0
        if self.transform:
            image = self.transform(image)
        return image
