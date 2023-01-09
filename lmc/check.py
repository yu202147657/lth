import torch
import numpy as np

if __name__ == "__main__":
    def check_dict(dict1_path, dict2_path):
        sd1 = torch.load(dict1_path)
        sd2 = torch.load(dict2_path)

        for ele in sd1.keys():
            #print(ele)
            #if ele == "bn.bias":
                #print(sd1[ele], '####', sd2[ele])
            print(sd1[ele] == sd2[ele])
        return

    check_dict('model_checkpoints/ckpts_pruning_CIFAR10_resnet20_ep_0_005_5_reg_None__sgd_cosine_lr_0_1_0_1_50_finetune_0_1_MAML_-1_10_fan_True_signed_constant_unif_width_1_0_seed_42_idx_None_replicate_1/model_ep120_it0.pt',
                    'model_checkpoints/ckpts_pruning_CIFAR10_resnet20_ep_0_005_5_reg_None__sgd_cosine_lr_0_1_0_1_50_finetune_0_1_MAML_-1_10_fan_True_signed_constant_unif_width_1_0_seed_42_idx_None_replicate_2/model_ep120_it0.pt')
