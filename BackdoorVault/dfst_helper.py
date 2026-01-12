"""
DFST (Deep Feature Space Trojan) Helper module.
Contains CycleGAN training functionality for DFST backdoor attack.
"""
import sys
import os
import torch
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from util import get_norm, get_size


def train_gan(attack, train_loader):
    """Train CycleGAN for DFST style transfer backdoor.
    
    Args:
        attack: Attack object containing DFST backdoor and optimizers
        train_loader: DataLoader for training data
    """
    print('-'*70)
    print('Training CycleGAN...')
    print('-'*70)
    attack.backdoor.genr_a2b.train()
    attack.backdoor.genr_b2a.train()
    attack.backdoor.disc_a.train()
    attack.backdoor.disc_b.train()

    normalize, unnormalize = attack.processing
    # Get target size from training data and create style transform
    img_size = get_size(attack.args.dataset)
    style_transform = transforms.Compose([
        transforms.Resize(img_size),
        normalize
    ])
    style_set = ImageDataset('data/sunrise', style_transform)
    style_loader = DataLoader(dataset=style_set, num_workers=4, shuffle=True,
                              batch_size=train_loader.batch_size)
    crit_mse = torch.nn.MSELoss()
    crit_l1  = torch.nn.L1Loss()

    for epoch in range(3000):
        for step, (x_a, x_b) in enumerate(zip(train_loader, style_loader)):
            x_a, x_b = x_a[0].to(attack.device), x_b.to(attack.device)
            size = min(x_a.size(0), x_b.size(0))
            x_a, x_b = x_a[:size], x_b[:size]

            # Get discriminator output shape dynamically
            with torch.no_grad():
                disc_out_shape = attack.backdoor.disc_a(x_a).shape[2:]  # Get spatial dims
            label_real = torch.ones( [size, 1, *disc_out_shape]).to(attack.device)
            label_fake = torch.zeros([size, 1, *disc_out_shape]).to(attack.device)

            # Generate images
            b_fake = normalize(attack.backdoor.genr_a2b(x_a))
            a_fake = normalize(attack.backdoor.genr_b2a(x_b))

            # Update discriminator
            attack.optim_disc_a.zero_grad()
            attack.optim_disc_b.zero_grad()

            loss_disc_a = crit_mse(attack.backdoor.disc_a(x_a), label_real) +\
                          crit_mse(attack.backdoor.disc_a(a_fake.detach()),
                                   label_fake)
            loss_disc_b = crit_mse(attack.backdoor.disc_b(x_b), label_real) +\
                          crit_mse(attack.backdoor.disc_b(b_fake.detach()),
                                   label_fake)
            loss_disc = loss_disc_a + loss_disc_b

            loss_disc_a.backward()
            attack.optim_disc_a.step()
            loss_disc_b.backward()
            attack.optim_disc_b.step()

            # Update generator
            attack.optim_genr_a2b.zero_grad()
            attack.optim_genr_b2a.zero_grad()

            loss_fool_da = crit_l1(attack.backdoor.disc_a(a_fake), label_real)
            loss_fool_db = crit_l1(attack.backdoor.disc_b(b_fake), label_real)
            loss_cycle_a = crit_l1(normalize(attack.backdoor.genr_b2a(b_fake)), x_a)
            loss_cycle_b = crit_l1(normalize(attack.backdoor.genr_a2b(a_fake)), x_b)
            loss_id_a2b  = crit_l1(normalize(attack.backdoor.genr_a2b(x_b)), x_b)
            loss_id_b2a  = crit_l1(normalize(attack.backdoor.genr_b2a(x_a)), x_a)

            loss_fool  = loss_fool_da + loss_fool_db
            loss_cycle = loss_cycle_a + loss_cycle_b
            loss_id    = loss_id_a2b  + loss_id_b2a
            loss_genr  = loss_fool + loss_cycle + loss_id

            loss_genr.backward()
            attack.optim_genr_a2b.step()
            attack.optim_genr_b2a.step()

            sys.stdout.write('\repoch {:4}, step: {:2}, [D loss: {:.4f}] '
                             .format(epoch, step, loss_disc) +\
                             '[G loss: {:.4f}, fool: {:.4f}, cycle: {:.4f} '
                             .format(loss_genr, loss_fool, loss_cycle) +\
                             'id: {:.4f}]'.format(loss_id))
            sys.stdout.flush()

        if epoch % 100 == 0:
            print()
            a_real = unnormalize(x_a)
            b_fake = unnormalize(b_fake)
            os.makedirs('data/sample', exist_ok=True)
            for i in range(min(10, size)):
                save_image(a_real[i], f'data/sample/gan_{i}_ori.png')
                save_image(b_fake[i], f'data/sample/gan_{i}_rec.png')

    attack.backdoor.genr_a2b.eval()
    attack.backdoor.genr_b2a.eval()
    attack.backdoor.disc_a.eval()
    attack.backdoor.disc_b.eval()
    attack.backdoor.genr_a2b.requires_grad_(False)
    attack.backdoor.genr_b2a.requires_grad_(False)
    attack.backdoor.disc_a.requires_grad_(False)
    attack.backdoor.disc_b.requires_grad_(False)
    print('-'*70)
