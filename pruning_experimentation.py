import models.registry, models.mnist_lenet
import pruning.sparse_global
import torch

import models.registry, models.mnist_lenet
import pruning.sparse_global
from pruning.mask import Mask
import platforms.registry
import platforms
from foundations import hparams
import datasets.registry
import torch

if __name__ == "__main__":
    hp = models.registry.get_default_hparams('mnist_lenet_300_100')
    model = models.registry.get(hp.model_hparams)
    model.load_state_dict(torch.load("open_lth_data/train_10908b66473bfc5cb7081ef43dee88a2/replicate_1/main"
                                     "/model_ep5_it0.pth"))

    # pruning_strategy = pruning.sparse_global.Strategy()
    # pruning_mask = pruning_strategy.prune(pruning_strategy.get_pruning_hparams(), model)
    # mask = Mask.ones_like(model).numpy()

    # pruned_weights = [model.state_dict()[k].numpy()[mask[k] == 1] for k in mask]
    # for k in mask:
    #     model.state_dict()[k].numpy()[mask[k] == 1] = 1

    plat = platforms.local.Platform()
    platforms.platform._PLATFORM = plat
    dataset_hparams = hparams.DatasetHparams("mnist", 50)
    dataloader = datasets.registry.get(dataset_hparams)
    train_features, train_labels = next(iter(dataloader))
    print(model.parameters())
    criterion = model.loss_criterion
    loss = criterion(model(train_features), train_labels)
    model.zero_grad()
    loss.backward()
    grad_dict = {k: v.grad**2 for k, v in model.named_parameters()}
    print(grad_dict)
    print(f"Loss: {loss}")