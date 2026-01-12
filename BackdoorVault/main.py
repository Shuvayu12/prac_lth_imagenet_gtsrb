# coding: utf-8
"""
DFST (Deep Feature Space Trojan) Backdoor Attack Main Script.
Simplified to only support DFST attack with ResNet18 models.
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
import numpy as np
import os
import sys
import time
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from attack import Attack
from dataset import PoisonDataset, ImageDataset
from dfst_helper import train_gan
from util import get_model, get_loader, get_dataset, get_norm, get_size, get_classes, get_backdoor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'


def eval_acc(model, loader):
    """Evaluate model accuracy on a data loader."""
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(x_batch)
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc
    

def train(args):
    """Train a clean ResNet18 model."""
    model = get_model(args.network, args.dataset).to(DEVICE)
    model = torch.nn.DataParallel(model)

    train_loader = get_loader(args, train=True)
    test_loader  = get_loader(args, train=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                                gamma=0.1)
    save_path = f'ckpt/{args.dataset}_{args.network}_clean.pt'

    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        time_end = time.time()
        acc = eval_acc(model, test_loader)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, '
                         .format(epoch, step, time_end-time_start) +\
                         'loss: {:.4f}, acc: {:.4f}\n'.format(loss, acc))
        sys.stdout.flush()
        time_start = time.time()

        torch.save(model, save_path)
        scheduler.step()


def test(args):
    """Test a trained model for accuracy and ASR."""
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.suffix}.pt'
    model = torch.load(model_filepath, map_location=DEVICE)
    model.eval()

    test_loader = get_loader(args, train=False)

    acc = eval_acc(model, test_loader)
    print(f'ACC: {acc:.4f}')

    # Test ASR for DFST
    test_set = get_dataset(args, train=False)
    shape = get_size(args.dataset)
    processing = get_norm(args.dataset)
    backdoor = get_backdoor(args.attack, shape, processing[0], DEVICE, args)
    
    poison_set = PoisonDataset(dataset=test_set, 
                               data_rate=1,
                               attack=args.attack, 
                               target=args.target,
                               poison_rate=1,
                               processing=processing, 
                               backdoor=backdoor)
    poison_loader = DataLoader(dataset=poison_set, num_workers=0,
                               batch_size=args.batch_size)
    asr = eval_acc(model, poison_loader)
    print(f'ASR: {asr:.4f}')


def poison(args):
    """Train a backdoored model using DFST attack."""
    model = get_model(args.network, args.dataset).to(DEVICE)
    model = torch.nn.DataParallel(model)

    attack = Attack(model, args, device=DEVICE)

    train_loader  = DataLoader(dataset=attack.train_set, num_workers=4,
                               batch_size=args.batch_size, shuffle=True)
    poison_loader = DataLoader(dataset=attack.poison_set, num_workers=0,
                               batch_size=args.batch_size)
    test_loader   = DataLoader(dataset=attack.test_set, num_workers=4,
                               batch_size=args.batch_size)

    save_path = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'

    # Train CycleGAN for DFST
    if args.attack == 'dfst':
        train_gan(attack, train_loader)
        torch.save(attack.backdoor.genr_a2b, f'{save_path[:-3]}_generator.pt')

    best_acc = 0
    best_asr = 0
    time_start = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            x_batch, y_batch = attack.inject(x_batch, y_batch)

            attack.optimizer.zero_grad()
            output = model(x_batch)
            loss = attack.criterion(output, y_batch)
            loss.backward()
            attack.optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        attack.scheduler.step()

        time_end = time.time()
        acc = eval_acc(model, test_loader)
        asr = eval_acc(model, poison_loader)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, acc: {:.4f}, '
                         .format(epoch, step, time_end-time_start, acc) +\
                         'asr: {:.4f}\n'.format(asr))
        sys.stdout.flush()
        time_start = time.time()

        if epoch > 10 and acc + asr > best_acc + best_asr:
            best_acc = acc
            best_asr = asr
            print(f'---BEST ACC: {best_acc:.4f}, ASR: {best_asr:.4f}---')
            torch.save(model, save_path)


def main():
    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
    elif args.phase == 'poison':
        poison(args)
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFST Backdoor Attack')

    parser.add_argument('--datadir', default='./data',    help='root directory of data')
    parser.add_argument('--suffix',  default='dfst',      help='suffix of saved path')
    parser.add_argument('--gpu',     default='0',         help='gpu id')

    parser.add_argument('--phase',   default='test',      help='phase: train, test, poison')
    parser.add_argument('--dataset', default='cifar10',   help='dataset: cifar10, gtsrb, tiny_imagenet')
    parser.add_argument('--network', default='resnet18',  help='network: resnet18')

    parser.add_argument('--attack',  default='dfst',      help='attack type: dfst')

    parser.add_argument('--seed',        type=int, default=1024, help='seed index')
    parser.add_argument('--batch_size',  type=int, default=128,  help='batch size')
    parser.add_argument('--epochs',      type=int, default=250,  help='number of epochs')
    parser.add_argument('--target',      type=int, default=0,    help='target label')

    parser.add_argument('--poison_rate', type=float, default=0.1,  help='poisoning rate')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda')

    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('='*50)
