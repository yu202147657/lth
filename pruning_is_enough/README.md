## Rare Gems Files 
Sourced from: https://github.com/ksreenivasan/pruning_is_enough

Models can be trained by:
```
python pruning_is_enough/main.py
```

The above code will train to default settings. To change the settings, the below are useful parameters:

```
python pruning_is_enough/main.py --config "<PATH TO CONFIG FILE>/resnet20_global_ep.yml" --algo "global_ep" --arch "resnet20"
```

Model hyperparameters are stored in the Config folder. 

Algorithims can be one of ```ep|pt_hack|pt_reg|hc|ep+greedy|greedy+ep|hc_iter|global_ep|global_ep_iter|imp```

## Notes from YN:
- I am not quite sure which is the correct EP ```algo``` to use
- There seem to be some issues with changing ```arch```. E.g. their code breaks if ```arch``` is set to ```resnet18```, even though this is a supported architecture in the config files.
- Path issues in their code with the non-default config files. (Currently path is hardcoded to the default - needs to be fixed)
