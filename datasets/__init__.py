from .cifar10 import CIFAR10

D_LIST = {
    "cifar10": CIFAR10, 
}

def prepare_dataset(args):
    dataset = D_LIST[args.dataset]
    train_set = dataset.build_dataset(train=True)
    test_set = dataset.build_dataset(train=False)
    return train_set, test_set