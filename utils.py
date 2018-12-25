import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path

def create_dataset(root_dir):
    ''' Create training Dataset of normalized images '''
    transforms = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Input image data is in range (0, 1), we normalize to be in range (-1, 1)
    return ImageFolder(root=root_dir, transform=transforms)
    
def init_params(module):
    ''' Initialize parameter values for Generator/Discriminator models '''
    classname = module.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
    elif 'Linear' in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        nn.init.constant_(module.bias.data, 0)