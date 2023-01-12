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

    check_dict('model_checkpoints/ckpts_pruning_CIFAR10_resnet20_hc_iter_0_1286398818033211_5_reg_L2_5e-05_sgd_cosine_lr_0_1_0_1_50_finetune_0_01_MAML_-1_10_fan_False_signed_constant_unif_width_1_0_seed_42_idx_None_1/model_ep0_it0.pt',
                    'model_checkpoints/ckpts_pruning_CIFAR10_resnet20_hc_iter_0_1286398818033211_5_reg_L2_5e-05_sgd_cosine_lr_0_1_0_1_50_finetune_0_01_MAML_-1_10_fan_False_signed_constant_unif_width_1_0_seed_42_idx_None/model_ep0_it0.pt')
