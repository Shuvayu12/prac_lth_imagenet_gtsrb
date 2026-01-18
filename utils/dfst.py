"""
DFST Utilities for PrAC-LTH.
Thin wrapper that imports from BackdoorVault and provides setup helpers.
Includes checkpoint conversion from BackdoorVault format to PrAC-LTH format.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from copy import deepcopy

# Add BackdoorVault to path FIRST (before other imports that might conflict)
_backdoor_vault_path = os.path.join(os.path.dirname(__file__), '..', 'BackdoorVault')
if _backdoor_vault_path not in sys.path:
    sys.path.insert(0, _backdoor_vault_path)

# Import from BackdoorVault explicitly
from BackdoorVault.dfst import DFST
from BackdoorVault.dataset import PoisonDataset
from BackdoorVault.util import GTSRBTestDataset


def get_dfst_paths(dataset):
    """Get the default DFST model and generator paths for a dataset."""
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'BackdoorVault', 'ckpt')
    model_paths = {
        'cifar10': os.path.join(base_dir, 'cifar10_resnet18_dfst.pt'),
        'gtsrb': os.path.join(base_dir, 'gtsrb_resnet18_dfst.pt'),
    }
    generator_paths = {
        'cifar10': os.path.join(base_dir, 'cifar10_resnet18_dfst_generator.pt'),
        'gtsrb': os.path.join(base_dir, 'gtsrb_resnet18_dfst_generator.pt'),
    }
    return model_paths.get(dataset), generator_paths.get(dataset)


def get_normalization(dataset):
    """Get normalization transforms for a dataset."""
    params = {
        'cifar10': ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        'cifar100': ([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]),
        'gtsrb': ([0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669]),
        'tiny-imagenet': ([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    }
    mean, std = params.get(dataset, params['cifar10'])
    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)
    normalize = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(-mean / std, 1 / std)
    return normalize, unnormalize


def setup_poison_loader(args, device):
    """Setup DFST backdoor and poison loader for ASR evaluation.
    
    Args:
        args: Command line arguments (needs dataset, data, batch_size)
        device: Torch device
        
    Returns:
        poison_loader: DataLoader for ASR evaluation
    """
    # Get normalization and generator path
    normalize, unnormalize = get_normalization(args.dataset)
    _, generator_path = get_dfst_paths(args.dataset)
    
    # Setup DFST backdoor
    backdoor = DFST(normalize, device=device)
    if generator_path and os.path.exists(generator_path):
        backdoor.genr_a2b = torch.load(generator_path, map_location=device, weights_only=False)
        if hasattr(backdoor.genr_a2b, 'to'):
            backdoor.genr_a2b = backdoor.genr_a2b.to(device)
        print(f'[DFST] Loaded generator: {generator_path}')
    else:
        print(f'[DFST] WARNING: Generator not found at {generator_path}')
    
    # Get test set based on dataset
    if args.dataset == 'cifar10':
        test_set = datasets.CIFAR10(root=args.data, train=False, download=False,
                                     transform=transforms.ToTensor())
    elif args.dataset == 'cifar100':
        test_set = datasets.CIFAR100(root=args.data, train=False, download=False,
                                      transform=transforms.ToTensor())
    elif args.dataset == 'gtsrb':
        # Use custom GTSRBTestDataset for flat structure with CSV labels
        test_path = os.path.join(args.data, 'GTSRB', 'Test')
        csv_file = os.path.join(test_path, 'GT-final_test.csv')
        # Try alternate CSV name if first doesn't exist
        if not os.path.exists(csv_file):
            csv_file = os.path.join(test_path, 'Test.csv')
        test_set = GTSRBTestDataset(
            root_dir=test_path,
            csv_file=csv_file,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
        )
    elif args.dataset == 'tiny-imagenet':
        test_set = datasets.ImageFolder(
            root=os.path.join(args.data, 'tiny-imagenet', 'val'),
            transform=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ]))
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    # Create poison dataset (100% poison rate for ASR evaluation)
    poison_dataset = PoisonDataset(
        dataset=test_set,
        target=0,  # Target class 0
        data_rate=1.0,
        poison_rate=1.0,
        processing=(normalize, unnormalize),
        backdoor=backdoor
    )
    
    poison_loader = torch.utils.data.DataLoader(
        poison_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    return poison_loader


def convert_dfst_to_prac_checkpoint(args, device):
    """Convert a BackdoorVault DFST model to PrAC-LTH checkpoint format.
    
    Args:
        args: Command line arguments (needs dataset, split_file, lr, momentum, weight_decay, decreasing_lr, queue_length)
        device: Torch device
        
    Returns:
        checkpoint: Dictionary in PrAC-LTH checkpoint format
    """
    from models.resnet_cifar import ResNet18_CIFAR
    from models.resnet_gtsrb import ResNet18_GTSRB
    from advertorch.utils import NormalizeByChannelMeanStd
    
    # Get DFST model path
    dfst_model_path, _ = get_dfst_paths(args.dataset)
    
    if dfst_model_path is None or not os.path.exists(dfst_model_path):
        raise FileNotFoundError(f'DFST model not found for dataset {args.dataset} at {dfst_model_path}')
    
    print(f'[DFST] Converting checkpoint: {dfst_model_path}')
    
    # Load DFST model (wrapped in DataParallel)
    dfst = torch.load(dfst_model_path, map_location=device, weights_only=False)
    
    # Unwrap DataParallel if needed
    if isinstance(dfst, nn.DataParallel):
        dfst_state = dfst.module.state_dict()
    else:
        dfst_state = dfst.state_dict()
    
    # Get normalization parameters and create target model
    norm_params = {
        'cifar10': ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], 10),
        'gtsrb': ([0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669], 43),
    }
    
    mean, std, num_classes = norm_params.get(args.dataset, norm_params['cifar10'])
    
    # Create target model
    if args.dataset == 'cifar10':
        target_model = ResNet18_CIFAR(num_classes=num_classes)
    elif args.dataset == 'gtsrb':
        target_model = ResNet18_GTSRB(num_classes=num_classes)
    else:
        target_model = ResNet18_CIFAR(num_classes=num_classes)
    
    # Add normalization as attribute (same as setup.py does)
    target_model.normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    
    # Use DFST state dict directly (same model structure from models/ folder)
    new_state_dict = dfst_state
    
    # Verify the state dict can be loaded
    try:
        target_model.load_state_dict(new_state_dict, strict=True)
        print(f'[DFST] Successfully loaded {len(new_state_dict)} parameters')
    except Exception as e:
        print(f'[DFST] Warning: Could not verify state dict: {e}')
    
    # Load training sequence
    train_number = {'cifar10': 45000, 'gtsrb': 35326}.get(args.dataset, 45000)
    sequence = np.load(args.split_file)[:train_number]
    
    # Create optimizer and scheduler for checkpoint
    optimizer = torch.optim.SGD(
        target_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    
    # Build checkpoint in PrAC-LTH format
    checkpoint = {
        'state': 0,
        'result': {'train': [], 'ta': [], 'test_ta': [], 'asr': []},
        'epoch': 0,
        'state_dict': new_state_dict,
        'best_sa': 0.0,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'init_weight': deepcopy(new_state_dict),
        'prediction': [],
        'sequence': sequence,
        'remain_para': sum(v.numel() for v in new_state_dict.values() if isinstance(v, torch.Tensor)),
        'distance_queue': torch.ones(args.queue_length),
        'last_mask': None,
        'start_record': False
    }
    
    print(f'[DFST] Checkpoint created with {checkpoint["remain_para"]} parameters')
    
    return checkpoint, target_model
