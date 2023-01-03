#Adapted from https://github.com/VITA-Group/Backdoor-LTH/blob/main/utils_linear_mode.py
import os 
import time
import copy 
import torch 
import numpy as np
import torch.nn.functional as F

def linear_interporation(state_dict1, state_dict2, alpha=1):

    new_dict = {}
    for key in state_dict1.keys():
        
        new_dict[key] = alpha * state_dict1[key] + (1 - alpha) * state_dict2[key]
    
    return new_dict


def linear_mode_connectivity(model, state_dict1, state_dict2, dataloader, batch_number=None, bins=10):
    
    original_weight = copy.deepcopy(model.state_dict())
    all_accuracy = []
    all_loss = []

    for i in range(bins+1):
        alpha = i/bins
        new_state_dict = linear_interporation(state_dict1, state_dict2, alpha)
        model.load_state_dict(new_state_dict)
        accuracy, loss = evaluation(dataloader, model, batch_number)
        all_accuracy.append(accuracy)
        all_loss.append(loss)
        print('alpha = {}, accuracy = {}, loss = {}'.format(alpha, accuracy, loss))

    # Accuracy
    top_acc = (all_accuracy[0] + all_accuracy[-1]) / 2
    bottom_acc = np.min(np.array(all_accuracy))

    # Loss
    top_loss = np.max(np.array(all_loss))
    bottom_loss = (all_loss[0] + all_loss[-1]) / 2 

    model.load_state_dict(original_weight)

    return 1-(top_acc - bottom_acc), top_loss - bottom_loss


def evaluation(dataloader, model, batch_number=None):

    model.eval()
    correct = 0
    number = 0

    for i, (input, target) in enumerate(dataloader):
        input = input.type(torch.FloatTensor)
        input = input.cuda()
        target = target.cuda()

        with torch.no_grad():
            output = model(image)
            predict = torch.argmax(output, 1)
            correct += (predict == target).float().sum().item()
            number += target.nelement() 

    acc = correct / number

    return acc
