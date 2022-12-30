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
from barrier import get_barrier


### RG and EP modules ##
from pruning_is_enough.main_utils import get_model
from pruning_is_enough.args_helper import parser_args


plat = platforms.local.Platform()
platforms.platform._PLATFORM = plat

def lmc(model_name, dataset_name, algo, dict1_path, dict2_path, batch_size=4):
    """Function takes 2 neural networks and linearly interpolates 
    between them. Returns loss and accuracy"""

    if dataset_name == 'cifar10':
        train_set = cifar10.Dataset.get_train_set(use_augmentation=False)
        test_set = cifar10.Dataset.get_test_set()
    else:
        train_set = mnist.Dataset.get_train_set(use_augmentation=False)
        test_set = mnist.Dataset.get_train_set(use_augmentation=False)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    if algo == 'IMP':
    
        #Need to add functionality for non-default hparams
        hp = models.registry.get_default_hparams(model_name)
        
        state_dict1, model = models.registry.load_model_and_dict(dict1_path, hp.model_hparams)
        state_dict2, model = models.registry.load_model_and_dict(dict2_path, hp.model_hparams)

    #else if algo = Rare Gems/EP
    else:
            parser_args.arch = model_name
            model = get_model(parser_args)
            
            state_dict1 = torch.load(dict1_path)
            state_dict2 = torch.load(dict2_path)

        
    acc, loss = linear_mode_connectivity(model, state_dict1, state_dict2, trainloader, batch_number=batch_size, bins=10)
    
    #A second implementation of error barrier
    #dict_before = get_barrier(model, state_dict1, state_dict2, trainloader, testloader)
    #barrier_train = dict_before['barrier_train']
    #lmc_train = dict_before['train_lmc']
    #print(barrier_train, 'barrier')
    #print(lmc_train, 'lmc')
    
    return acc, loss



if __name__ == "__main__":
    
    #Tested functions with 2 identical networks 
    #Accuracy is close to 100%, and loss close to 0 - as makes sense
    
    acc, loss = lmc('mnist_lenet_300_100', 'mnist', 'IMP', 
            'open_lth_data/train_10908b66473bfc5cb7081ef43dee88a2/replicate_1/main/model_ep5_it0.pth',
            'open_lth_data/train_10908b66473bfc5cb7081ef43dee88a2/replicate_1/main/model_ep5_it0.pth')

    print(acc, loss)
