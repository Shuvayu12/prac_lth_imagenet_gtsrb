"""
Attack module for DFST (Deep Feature Space Trojan) backdoor attack.
Simplified to only support DFST attack.
"""
import os
import torch
from backdoors import DFST
from dataset import PoisonDataset
from util import get_size, get_norm, get_dataset


class Attack:
    """DFST Attack class for backdoor injection."""
    
    def __init__(self, model, args, device=None):
        self.device = device
        self.attack = args.attack
        self.target = args.target
        self.poison_rate = args.poison_rate

        self.shape = get_size(args.dataset)
        self.processing = get_norm(args.dataset)
        
        # Initialize DFST backdoor
        self.backdoor = DFST(self.processing[0], device=self.device)
        
        # Load generator if exists
        base_path = f'ckpt/{args.dataset}_{args.network}'
        genr_path = f'{base_path}_dfst_generator.pt'
        if os.path.exists(genr_path):
            self.backdoor.genr_a2b = torch.load(genr_path, map_location=self.device)

        # Setup optimizers for DFST CycleGAN components
        self.optim_genr_a2b = torch.optim.Adam(
            self.backdoor.genr_a2b.parameters(),
            2e-4, betas=(0.5, 0.999))
        self.optim_genr_b2a = torch.optim.Adam(
            self.backdoor.genr_b2a.parameters(),
            2e-4, betas=(0.5, 0.999))
        self.optim_disc_a = torch.optim.Adam(
            self.backdoor.disc_a.parameters(),
            2e-4, betas=(0.5, 0.999))
        self.optim_disc_b = torch.optim.Adam(
            self.backdoor.disc_b.parameters(),
            2e-4, betas=(0.5, 0.999))

        # Standard training setup
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-1,
                                         momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=50, gamma=0.1)

        # Datasets
        self.train_set = get_dataset(args, train=True)
        self.test_set = get_dataset(args, train=False)
        
        # Poison test set for evaluation
        self.poison_set = PoisonDataset(
            dataset=self.test_set, 
            data_rate=1,
            attack=self.attack,
            target=self.target, 
            poison_rate=1,
            processing=self.processing,
            backdoor=self.backdoor
        )

    def inject(self, inputs, labels):
        """Inject DFST backdoor into a batch of inputs.
        
        Args:
            inputs: Input batch tensor
            labels: Label batch tensor
            
        Returns:
            Tuple of (poisoned_inputs, modified_labels)
        """
        num_bd = int(inputs.size(0) * self.poison_rate)
        
        # Apply DFST style transfer to poisoned samples
        inputs_bd = self.backdoor.inject(inputs[:num_bd])
        labels_bd = torch.full((num_bd,), self.target).to(self.device)

        # Concatenate poisoned and clean samples
        inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
        labels = torch.cat([labels_bd, labels[num_bd:]], dim=0)

        return inputs, labels
