"""
Dataset module for DFST backdoor attack.
Contains PoisonDataset and ImageDataset for DFST style transfer attack.
"""
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
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


class CachedPoisonDataset(Dataset):
    """Pre-cached poisoned dataset for fast ASR evaluation.
    
    Generates all poisoned images once and stores them in memory,
    avoiding expensive on-the-fly CycleGAN inference during each evaluation.
    """
    
    def __init__(self, dataset, target, backdoor, device, batch_size=64, max_samples=None):
        """
        Args:
            dataset: Base dataset to poison
            target: Target label for poisoned samples
            backdoor: DFST backdoor instance
            device: Torch device
            batch_size: Batch size for generating poisoned images
            max_samples: Maximum number of samples to cache (None = all)
        """
        print("Pre-caching poisoned images for ASR evaluation...")
        self.target = target
        
        # Use a temporary DataLoader to iterate through the dataset efficiently
        temp_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,  # Use 0 for compatibility
            shuffle=False
        )
        
        # Generate poisoned images in batches, filtering out target-labeled samples
        all_images = []
        all_labels = []
        total_cached = 0
        
        backdoor.genr_a2b.eval()
        with torch.no_grad():
            for batch_idx, (imgs, lbls) in enumerate(temp_loader):
                # Filter out samples with target label
                mask = lbls != target
                if mask.sum() == 0:
                    continue
                    
                imgs = imgs[mask].to(device)
                poisoned = backdoor.inject(imgs)
                all_images.append(poisoned.cpu())
                total_cached += imgs.size(0)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Cached {total_cached} images...")
                
                if max_samples and total_cached >= max_samples:
                    break
        
        if len(all_images) == 0:
            raise RuntimeError(f"No images found to poison (all have target label {target}?)")
        
        self.images = torch.cat(all_images, dim=0)
        if max_samples:
            self.images = self.images[:max_samples]
        self.labels = torch.full((len(self.images),), target, dtype=torch.long)
        print(f"Cached {len(self.images)} poisoned images for ASR evaluation")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
