import torch
import numpy as np

import models.registry, models.mnist_lenet
import platforms.registry
import platforms
from foundations import hparams
from foundations.step import Step
import datasets.registry
from datasets import mnist, cifar10

from utils_linear_mode import linear_mode_connectivity

plat = platforms.local.Platform()
platforms.platform._PLATFORM = plat

def lmc(model_name, dataset_name, algo, dict1_path, dict2_path, batch_size=4):
    """Function takes 2 neural networks and linearly interpolates 
    between them. Returns loss and accuracy"""

    if dataset_name == 'cifar10':
        train_set = CIFAR10.get_train_set(use_augmentation=False)
    else:
        train_set = mnist.Dataset.get_train_set(use_augmentation=False)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    if algo == 'IMP':
    
        #Need to add functionality for non-default hparams
        hp = models.registry.get_default_hparams(model_name)
        
        state_dict1, model = models.registry.load_model_and_dict(dict1_path, hp.model_hparams)
        state_dict2, model = models.registry.load_model_and_dict(dict2_path, hp.model_hparams)
    
    elif algo == 'Rare Gems':
        print('Work In Progress')
    else:
        ('EP work in progress')
        
    acc, loss = linear_mode_connectivity(model, state_dict1, state_dict2, trainloader, batch_number=batch_size, bins=10)

    return acc, loss



if __name__ == "__main__":
    
    #Tested functions with 2 identical networks 
    #Accuracy is close to 100%, and loss close to 0 - as makes sense
    acc, loss = lmc('mnist_lenet_300_100', 'mnist', 'IMP', 
                'lth/open_lth_data/train_10908b66473bfc5cb7081ef43dee88a2/replicate_1/main/model_ep5_it0.pth',
                'lth/open_lth_data/train_10908b66473bfc5cb7081ef43dee88a2/replicate_1/main/model_ep5_it0.pth')

    print(acc, loss)
