import models.registry, models.mnist_lenet
import pruning.sparse_global
import torch

if __name__ == "__main__":
    hp = models.registry.get_default_hparams('mnist_lenet_300_100')
    model = models.registry.get(hp.model_hparams)
    model.load_state_dict(torch.load("open_lth_data/train_10908b66473bfc5cb7081ef43dee88a2/replicate_1/main"
                                     "/model_ep5_it0.pth"))

    pruning_strategy = pruning.sparse_global.Strategy()
    pruning_mask = pruning_strategy.prune(pruning_strategy.get_pruning_hparams(), model)
    print(f"Mask: {pruning_mask}")
