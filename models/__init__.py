# External ResNet models for different datasets
from .resnet_cifar import ResNet18_CIFAR
from .resnet_gtsrb import ResNet18_GTSRB
from .resnet_tiny_imagenet import ResNet18_TinyImageNet

__all__ = ['ResNet18_CIFAR', 'ResNet18_GTSRB', 'ResNet18_TinyImageNet']
