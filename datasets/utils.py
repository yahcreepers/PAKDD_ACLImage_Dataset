from .cifar10 import CIFAR10
from .cifar20 import CIFAR20
from .cifar100 import CIFAR100
from .min10 import Micro_ImageNet10
from .min20 import Micro_ImageNet20

D_LIST = {
    "cifar10": CIFAR10, 
    "cifar20": CIFAR20,
    "cifar100": CIFAR100,  
    "min10": Micro_ImageNet10, 
    "min20": Micro_ImageNet20, 
    "clcifar10": CIFAR10, 
    "clcifar20": CIFAR20, 
    "clmin10": Micro_ImageNet10, 
    "clmin20": Micro_ImageNet20, 
    "cifar10n": CIFAR10, 
    "cifar20n": CIFAR20, 
}

def prepare_dataset(args):
    dataset = D_LIST[args.dataset]
    train_set = dataset.build_dataset(train=True, long_label=args.long_label, do_transform=args.do_transform)
    test_set = dataset.build_dataset(train=False, long_label=args.long_label, do_transform=args.do_transform)
    return train_set, test_set