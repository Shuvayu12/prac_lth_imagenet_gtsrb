"""
DFST (Deep Feature Space Trojan) Backdoor Attack Module.
Uses CycleGAN for style transfer based backdoor injection.
"""
import torch
from models.cyclegan import CycleGenerator, CycleDiscriminator


class DFST:
    """DFST backdoor attack using CycleGAN style transfer."""
    
    def __init__(self, normalize, device=None):
        """
        Args:
            normalize: Normalization transform for images
            device: Torch device (cuda/cpu)
        """
        self.device = device
        self.normalize = normalize

        # CycleGAN components
        self.genr_a2b = CycleGenerator().to(self.device)
        self.genr_b2a = CycleGenerator().to(self.device)
        self.disc_a = CycleDiscriminator().to(self.device)
        self.disc_b = CycleDiscriminator().to(self.device)
        
        # Wrap with DataParallel for multi-GPU support
        self.genr_a2b = torch.nn.DataParallel(self.genr_a2b)
        self.genr_b2a = torch.nn.DataParallel(self.genr_b2a)
        self.disc_a = torch.nn.DataParallel(self.disc_a)
        self.disc_b = torch.nn.DataParallel(self.disc_b)

    def inject(self, inputs):
        """Inject DFST backdoor by applying style transfer.
        
        Args:
            inputs: Input tensor batch
            
        Returns:
            Normalized style-transferred images
        """
        return self.normalize(self.genr_a2b(inputs))
