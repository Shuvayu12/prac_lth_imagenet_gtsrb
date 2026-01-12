# BackdoorVault - DFST Only

This is a simplified version of BackdoorVault that focuses exclusively on the **DFST (Deep Feature Space Trojan)** backdoor attack.

The DFST attack uses CycleGAN-based style transfer to inject backdoors into deep neural networks.

## Structure

```
BackdoorVault/
├── attack.py           # DFST attack implementation
├── backdoors/
│   ├── __init__.py
│   └── dfst.py         # DFST backdoor class using CycleGAN
├── dataset.py          # PoisonDataset and ImageDataset for DFST
├── helper/
│   ├── __init__.py
│   └── dfst_helper.py  # CycleGAN training utilities
├── main.py             # Main entry point for DFST experiments
├── models/
│   ├── __init__.py
│   └── cyclegan.py     # CycleGAN generator and discriminator
├── util.py             # Utilities and external ResNet model loading
└── data/
    └── sunrise/        # Style images for DFST transfer
```

## External Model Integration

This version uses ResNet18 models from the parent `models/` directory:
- `ResNet18_CIFAR` - For CIFAR-10 dataset (32x32 images, 10 classes)
- `ResNet18_GTSRB` - For GTSRB dataset (32x32 images, 43 classes)
- `ResNet18_TinyImageNet` - For Tiny ImageNet dataset (64x64 images, 200 classes)

## Usage

### Train a clean model:
```bash
python main.py --phase train --dataset cifar10 --network resnet18 --epochs 200
```

### Train DFST backdoored model:
```bash
python main.py --phase poison --dataset cifar10 --network resnet18 --attack dfst --target 0 --poison_rate 0.1 --epochs 250
```

### Test a trained model:
```bash
python main.py --phase test --dataset cifar10 --network resnet18 --suffix dfst --attack dfst
```

## Supported Datasets
- CIFAR-10 (`--dataset cifar10`)
- GTSRB (`--dataset gtsrb`)
- Tiny ImageNet (`--dataset tiny_imagenet`)

## Arguments
- `--datadir`: Root directory for datasets (default: `./data`)
- `--phase`: Phase to run (`train`, `test`, `poison`)
- `--dataset`: Dataset name
- `--network`: Network architecture (only `resnet18` supported)
- `--attack`: Attack type (only `dfst` supported)
- `--target`: Target label for backdoor (default: 0)
- `--poison_rate`: Fraction of training data to poison (default: 0.1)
- `--epochs`: Number of training epochs (default: 250)
- `--batch_size`: Batch size (default: 128)
- `--gpu`: GPU ID (default: '0')
- `--seed`: Random seed (default: 1024)

## Reference

DFST: [Deep Feature Space Trojan Attack](https://arxiv.org/abs/2012.11212)

Based on [BackdoorVault](https://github.com/taog0/BackdoorVault) toolbox.
