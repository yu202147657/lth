from pruning_is_enough.models.resnet import ResNet18, ResNet50, ResNet101, WideResNet50_2, WideResNet101_2
from pruning_is_enough.models.resnet_kaiming import resnet20, resnet32, resnet32_double
from pruning_is_enough.models.mobilenet import MobileNetV2
from pruning_is_enough.models.frankle import FC, Conv2, Conv4, Conv4Normal, Conv6, Conv4Wide, Conv8, Conv6Wide
from pruning_is_enough.models.wideresnet import WideResNet28
from pruning_is_enough.models.vgg import vgg16, tinyvgg16

#### TODO: delete below ones (merge with above code)
from pruning_is_enough.models.resnet_cifar import cResNet18, cResNet50 
from pruning_is_enough.models.resnet_tiny import TinyResNet18 

__all__ = [
    "tinyvgg16",
    "vgg16",
    "ResNet18",
    "ResNet50",
    "ResNet101",
    "WideResNet50_2",
    "WideResNet101_2",
    "resnet20",
    "resnet32",
    "resnet32_double",
    "MobileNetV2",
    "FC",
    "Conv2",
    "Conv4",
    "Conv4Normal",
    "Conv6",
    "Conv4Wide",
    "Conv8",
    "Conv6Wide",
    "cResNet18",
    "cResNet50",
    "TinyResNet18",
    "WideResNet28"
]
