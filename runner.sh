python open_lth.py lottery --default_hparams cifar_resnet_20 --level=3 --rewind=500it
python open_lth.py lottery --default_hparams cifar_resnet_20 --level=3 --rewind=500it --replicate=2
python open_lth.py train --default_hparams cifar_resnet_20
python open_lth.py train --default_hparams cifar_resnet_20 --replicate=2
python open_lth.py train --default_hparams mnist_lenet_300_100
python open_lth.py train --default_hparams mnist_lenet_300_100 --replicate=2