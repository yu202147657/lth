from lmc.main_lmc import lmc

###################### IMP  ################################
# python3 open_lth.py train --default_hparams <mnist_lenet_300_100, cifar_resnet_20> 
#MNIST LENET
barrier_train = lmc('mnist_lenet_300_100', 'mnist', 'IMP',
                    'open_lth_data/train_574e51abc295d8da78175b320504f2ba/replicate_1/main/model_ep40_it0.pth',
                    'open_lth_data/train_574e51abc295d8da78175b320504f2ba/replicate_2/main/model_ep40_it0.pth')

#CIFAR RESNET
barrier_train = lmc('cifar_resnet_20', 'cifar10', 'IMP',
                    'open_lth_data/train_71bc92a970b64a76d7ab7681764b0021/replicate_1/main/model_ep160_it0.pth',
                    'open_lth_data/train_71bc92a970b64a76d7ab7681764b0021/replicate_2/main/model_ep160_it0.pth')

#CIFAR RESNET- Rewind
#python open_lth.py lottery --default_hparams cifar_resnet_20 --rewind=500it --levels=3
#python open_lth.py lottery --default_hparams cifar_resnet_20 --rewind=500it --levels=8 --training_steps=630000it --milestone_steps=32000it,48000it
barrier_train = lmc('cifar_resnet_20', 'cifar10', 'IMP',
                    'open_lth_data/lottery_23b644efaef60c49ca88fc5e37e2595a/replicate_1/level_8/main/model_ep160_it0.pth',
                    'open_lth_data/lottery_23b644efaef60c49ca88fc5e37e2595a/replicate_2/level_8/main/model_ep160_it0.pth')

###################### Rare Gems  ##########################
#python pruning_is_enough/main.py --fixed-init

#CIFAR RESNET
barrier_train = lmc('resnet20', 'cifar10', 'GM',
                        'model_checkpoints/ckpts_pruning_CIFAR10_resnet20_hc_iter_0_1286398818033211_5_reg_L2_5e-05_sgd_cosine_lr_0_1_0_1_50_finetune_0_01_MAML_-1_10_fan_False_signed_constant_unif_width_1_0_seed_42_idx_None_replicate_1/model_ep155_it0.pt',
                        'model_checkpoints/ckpts_pruning_CIFAR10_resnet20_hc_iter_0_1286398818033211_5_reg_L2_5e-05_sgd_cosine_lr_0_1_0_1_50_finetune_0_01_MAML_-1_10_fan_False_signed_constant_unif_width_1_0_seed_42_idx_None_replicate_2/model_ep155_it0.pt')
    
    
###################### EP  ################################
#python pruning_is_enough/main.py --config "pruning_is_enough/configs/ablation_ep_gm_resnet20_059/ep.yml" --algo "ep" --arch "resnet20" --fixed-init

#could only train for 120 epochs ea/

#CIFAR RESNET
barrier_train = lmc('resnet20', 'cifar10', 'EP', 'model_checkpoints/ckpts_pruning_CIFAR10_resnet20_ep_0_005_5_reg_None__sgd_cosine_lr_0_1_0_1_50_finetune_0_1_MAML_-1_10_fan_True_signed_constant_unif_width_1_0_seed_42_idx_None_replicate_1/model_ep120_it0.pt',
                    'model_checkpoints/ckpts_pruning_CIFAR10_resnet20_ep_0_005_5_reg_None__sgd_cosine_lr_0_1_0_1_50_finetune_0_1_MAML_-1_10_fan_True_signed_constant_unif_width_1_0_seed_42_idx_None_replicate_2/model_ep120_it0.pt')