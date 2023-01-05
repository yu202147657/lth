# Adapted from https://github.com/rahimentezari/PermutationInvariance/blob/main/barrier.py
import numpy as np
import torch
import csv


# Calculate error barrier
def get_barrier(model, sd1, sd2, trainloader, testloader, filename):
    barrier_dict = {}

    weights = np.linspace(0, 1, 11)
    test_errors = []
    train_errors = []

    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')

        for i in range(len(weights)):
            model.load_state_dict(interpolate_state_dicts(sd1, sd2, weights[i]))
            train_errors.append(get_error(model, trainloader))
            test_errors.append(get_error(model, testloader))
            writer.writerow([weights[i], train_errors[i]])

    model1_test_error = test_errors[0]
    model2_test_error = test_errors[-1]
    test_avg_models = (model1_test_error + model2_test_error) / 2

    model1_train_error = train_errors[0]
    model2_train_error = train_errors[-1]
    train_avg_models = (model1_train_error + model2_train_error) / 2

    print(f"Train Errors: {train_errors}")
    train_error_barrier = max(train_errors) - train_avg_models
    print(f"Train Br: {train_error_barrier}")
    test_error_barrier = max(test_errors) - test_avg_models
    barrier_dict['barrier_test'] = test_error_barrier
    barrier_dict['barrier_train'] = train_error_barrier

    return barrier_dict


def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
    return {key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key]
            for key in state_dict_1.keys()}


def get_error(model, dataloader):
    #if torch.cuda.is_available():
    #    device = torch.device('cuda')
    #else:
    #    device = torch.device('cpu')
    device = torch.device('cpu')
    model.to(device)

    # switch to evaluate mode
    model.eval()
    num_correct = 0.0
    num_samples = 0.0

    with torch.no_grad():
        i = 0
        for input, label in dataloader:
            preds = np.argmax(model(input.cpu()), axis=1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
            i += 1


    return float(100 * (1 - num_correct / num_samples))
