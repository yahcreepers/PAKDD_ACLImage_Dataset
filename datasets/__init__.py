from .cifar10 import CIFAR10
from .min10 import Micro_ImageNet10

D_LIST = {
    "cifar10": CIFAR10, 
    "min10": Micro_ImageNet10, 
}

def prepare_dataset(args):
    dataset = D_LIST[args.dataset]
    train_set = dataset.build_dataset(train=True)
    test_set = dataset.build_dataset(train=False)
    return train_set, test_set