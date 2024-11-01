from .cifar10 import CIFAR10
from .cifar20 import CIFAR20
from .min10 import Micro_ImageNet10
from .min20 import Micro_ImageNet20

D_LIST = {
    "cifar10": CIFAR10, 
    "cifar20": CIFAR20, 
    "min10": Micro_ImageNet10, 
    "min20": Micro_ImageNet20, 
}

def prepare_dataset(args):
    dataset = D_LIST[args.dataset]
    train_set = dataset.build_dataset(train=True, long_label=args.long_label)
    test_set = dataset.build_dataset(train=False, long_label=args.long_label)
    return train_set, test_set