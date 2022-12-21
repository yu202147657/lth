import numpy as np

import models.registry, models.mnist_lenet
import pruning.fisher_pruning
from pruning.mask import Mask
import platforms.registry
import platforms
from foundations import hparams
import datasets.registry
import torch

if __name__ == "__main__":
    plat = platforms.local.Platform()
    platforms.platform._PLATFORM = plat
    hp = models.registry.get_default_hparams('mnist_lenet_300_100')
    trained_model = models.registry.get(hp.model_hparams)
    trained_model.load_state_dict(torch.load("open_lth_data/train_10908b66473bfc5cb7081ef43dee88a2/replicate_1/main"
                                     "/model_ep5_it0.pth"))

    prunable_tensors = set(trained_model.prunable_layer_names)
    weights = {k: v.clone().cpu().detach().numpy()
               for k, v in trained_model.state_dict().items()
               if k in prunable_tensors}

    pruner = pruning.fisher_pruning.FisherStrategy()
    pruning_mask = pruner.prune(pruner.get_pruning_hparams(), trained_model)
    print(f"Mask: {pruning_mask}")
    total_ones = 0
    total_items = 0
    for k, v in pruning_mask.items():
        total_items += torch.numel(v)
        total_ones += torch.sum(v)
    print(total_ones/total_items)

    # pruning_strategy = pruning.sparse_global.Strategy()
    # pruning_mask = pruning_strategy.prune(pruning_strategy.get_pruning_hparams(), model)
    # mask = Mask.ones_like(model).numpy()

    # pruned_weights = [model.state_dict()[k].numpy()[mask[k] == 1] for k in mask]
    # for k in mask:
    #     model.state_dict()[k].numpy()[mask[k] == 1] = 1


    """
    NOTE: Code in this multi-line comment was used to test Fisher pruning
    plat = platforms.local.Platform()
    platforms.platform._PLATFORM = plat

    num_datapoints = 128
    dataset_hparams = hparams.DatasetHparams("mnist", 1)
    dataloader = datasets.registry.get(dataset_hparams)
    dataloader_iterator = iter(dataloader)

    squared_gradients_accumulator = {}
    prunable_tensors = set(trained_model.prunable_layer_names)
    for name in prunable_tensors:
        squared_gradients_accumulator[name] = np.zeros(list(trained_model.state_dict()[name].shape))

    dataloader_iterator = iter(dataloader)
    for _ in range(num_datapoints):
        train_features, train_labels = next(dataloader_iterator)
        criterion = trained_model.loss_criterion
        loss = criterion(trained_model(train_features), train_labels)
        trained_model.zero_grad()
        loss.backward()
        grad_dict = {k: v.grad.clone().cpu().detach().numpy() ** 2 for k, v in trained_model.named_parameters() if k in prunable_tensors}
        for k in grad_dict.keys():
            squared_gradients_accumulator[k] += grad_dict[k]


    weights = {k: v.clone().cpu().detach().numpy()
               for k, v in trained_model.state_dict().items()
               if k in prunable_tensors}
    weights_squared = {k: v.clone().cpu().detach().numpy() ** 2
                       for k, v in trained_model.state_dict().items()
                       if k in prunable_tensors}

    delta_loss = {}
    for k in prunable_tensors:
        delta_loss[k] = (1 / (2 * num_datapoints)) * np.multiply(weights_squared[k],
                                                                                  squared_gradients_accumulator[k])


    current_mask = Mask.ones_like(trained_model).numpy()
    # print(f"{delta_loss.items().shape()}"
    delta_loss_vector = np.concatenate([v[current_mask[k] == 1] for k, v in delta_loss.items()])
    threshold = np.sort(delta_loss_vector)[10000]
    print(f"Threshold: {threshold}")
    print(f"Delta Loss: {delta_loss}")
    new_mask = Mask({k: np.where(v > threshold, current_mask[k], np.zeros_like(v))
                     for k, v in delta_loss.items()})
    print(f"New Mask: {new_mask}")
    """

    # print(f"Delta Loss: {delta_loss}")
    # print(f"New Mask: {new_mask}")

    # print(f"Weights: {weights}")
    # print(f"Squared Weights: {weights_squared}")
    # print(f"Accumulated Gradients: {squared_gradients_accumulator.keys()}")


    # print(f"SQUARE: {squared_gradients_accumulator}")
    # for _ in range(50):
    #     train_features, train_labels = next(dataloader_iterator)
    #     print(f"Label: {train_labels}")
    # print(model.parameters())
    # criterion = model.loss_criterion
    # loss = criterion(model(train_features), train_labels)
    # model.zero_grad()
    # loss.backward()
    # print(prunable_tensors)
    # grad_dict = {k: v.grad**2 for k, v in model.named_parameters() if k in prunable_tensors}
    # for k in grad_dict.keys():
    #     squared_gradients_accumulator[k] += grad_dict[k]
    #     squared_gradients_accumulator[k] += grad_dict[k]
    #
    #
    # print(grad_dict)
    # print(squared_gradients_accumulator)
    # 0.0000e+00, 3.0593e-12, 0.0000e+00, 2.8477e-11
    # print(f"Loss: {loss}")
