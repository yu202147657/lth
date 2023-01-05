import torch
import numpy as np

import models.registry, models.mnist_lenet
import platforms.registry
import platforms
from foundations import hparams
from foundations.step import Step
import datasets.registry
from datasets import mnist, cifar10

from lmc.barrier import get_barrier

### RG and EP modules ##
from pruning_is_enough.main_utils import get_model
from pruning_is_enough.args_helper import parser_args

plat = platforms.local.Platform()
platforms.platform._PLATFORM = plat


def lmc(model_name, dataset_name, algo, dict1_path, dict2_path, batch_size=512):
    """Function takes 2 neural networks and linearly interpolates 
    between them"""

    if dataset_name == 'cifar10':
        train_set = cifar10.Dataset.get_train_set(use_augmentation=False)
        test_set = cifar10.Dataset.get_test_set()
    else:
        train_set = mnist.Dataset.get_train_set(use_augmentation=False)
        test_set = mnist.Dataset.get_test_set()

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    if algo == 'IMP':
    
        #Need to add functionality for non-default hparams
        hp = models.registry.get_default_hparams(model_name)
        
        state_dict1, model = models.registry.load_model_and_dict(dict1_path, hp.model_hparams)
        state_dict2, model = models.registry.load_model_and_dict(dict2_path, hp.model_hparams)

    #else if algo = GM/EP
    else:
        parser_args.arch = model_name
        model = get_model(parser_args)

        state_dict1 = torch.load(dict1_path)
        state_dict2 = torch.load(dict2_path)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)

    filename = 'lmc/experiments/' + f'{model_name}_{algo}.csv'

    # Error barrier calculation
    accuracy_dict = get_barrier(model, state_dict1, state_dict2, trainloader, testloader, filename)
    barrier_train = accuracy_dict['barrier_train']

    return barrier_train


