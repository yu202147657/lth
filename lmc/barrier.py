#Adapted from https://github.com/rahimentezari/PermutationInvariance/blob/main/barrier.py
import numpy as np
import torch

#Calculate error barrier
def get_barrier(model, sd1, sd2, trainloader, testloader):
    dict_barrier = {}
    ####################### get the barrier - before permutation
    ###### LMC
    weights = np.linspace(0, 1, 11)
    result_test = []
    result_train = []
    for i in range(len(weights)):
        model.load_state_dict(interpolate_state_dicts(sd1, sd2, weights[i]))
        result_train.append(evaluate_model(model, trainloader)['top1'])
        result_test.append(evaluate_model(model, testloader)['top1'])

    model1_eval = result_test[0]
    model2_eval = result_test[-1]
    test_avg_models = (model1_eval + model2_eval) / 2

    model1_eval = result_train[0]
    model2_eval = result_train[-1]
    train_avg_models = (model1_eval + model2_eval) / 2

    add_element(dict_barrier, 'train_avg_models', train_avg_models)
    add_element(dict_barrier, 'test_avg_models', test_avg_models)
    add_element(dict_barrier, 'train_lmc', result_train)
    add_element(dict_barrier, 'test_lmc', result_test)
    add_element(dict_barrier, 'barrier_test', test_avg_models - result_test[5])
    add_element(dict_barrier, 'barrier_train', train_avg_models - result_train[5])

    return dict_barrier


def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
    return {key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key]
            for key in state_dict_1.keys()}


def evaluate_model(model, dataloader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        
        for i, (input, target) in enumerate(dataloader):
            input = input.type(torch.FloatTensor)
            #input = data.to(device)
            #target = target.to(device)
            # compute output
            output = model(input)

            # measure accuracy and record loss
            acc1 = calc_accuracy(output, target, topk=(1,))[0]
            top1.update(acc1[0], input.size(0))
            # break
        # results = dict(top1=top1.avg, loss=losses.avg, batch_time=batch_time.avg)
        results = dict(top1=top1.avg)

    return {key: float(val) for key, val in results.items()}



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


def calc_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)
    
    


