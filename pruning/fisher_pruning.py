# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
import datasets.registry
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class FisherPruningHparams(hparams.PruningHparams):
    pruning_strategy = 'fisher_pruning'
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None
    dataset_name: str = "mnist"
    num_data_points: int = 128

    _name = 'Hyperparameters for Fisher Pruning Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'
    _dataset_name = 'Dataset to use for computing Fisher information calculation'
    _num_data_points = 'Number of data points to use to calculate Fisher information'


class FisherStrategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return FisherPruningHparams

    @staticmethod
    def prune(pruning_hparams: FisherPruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        dataset_hparams = hparams.DatasetHparams(pruning_hparams.dataset_name, 1)
        dataloader = datasets.registry.get(dataset_hparams)

        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Dictionary to store sum of squares of loss derivative w.r.t each parameter/training point
        squared_gradients_accumulator = {}
        for name in prunable_tensors:
            squared_gradients_accumulator[name] = np.zeros(list(trained_model.state_dict()[name].shape))

        # Loop over dataset, computing the square of the derivative of loss function w.r.t each prunable parameter
        dataloader_iterator = iter(dataloader)
        for _ in range(pruning_hparams.num_data_points):
            train_features, train_labels = next(dataloader_iterator)
            criterion = trained_model.loss_criterion
            loss = criterion(trained_model(train_features), train_labels)
            trained_model.zero_grad()
            loss.backward()
            # TODO: Double check detaching etc. doesn't cause any changes
            grad_dict = {k: v.grad.clone().cpu().detach().numpy() ** 2 for k, v in trained_model.named_parameters() if k in prunable_tensors}
            for k in grad_dict.keys():
                squared_gradients_accumulator[k] += grad_dict[k]

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        # Get the square of the model weights
        weights_squared = {k: v.clone().cpu().detach().numpy() ** 2
                           for k, v in trained_model.state_dict().items()
                           if k in prunable_tensors}

        # Calculate approximate increase in loss corresponding to pruning each parameter individually
        delta_loss = {}
        for k in prunable_tensors:
            delta_loss[k] = (1 / (2 * pruning_hparams.num_data_points)) * np.multiply(weights_squared[k],
                                                                                      squared_gradients_accumulator[k])


        delta_loss_vector = np.concatenate([v[current_mask[k] == 1] for k, v in delta_loss.items()])
        threshold = np.sort(delta_loss_vector)[number_of_weights_to_prune]

        # Keep weights corresponding to larger increase in loss
        new_mask = Mask({k: np.where(v > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in delta_loss.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
