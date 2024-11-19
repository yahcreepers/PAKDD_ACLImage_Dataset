import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import urllib.request
from tqdm import tqdm
import pickle
import gdown
import os
from .randomaugment import RandomAugment

class CIFAR100(torchvision.datasets.CIFAR100):
    """

    Real-world complementary-label dataset. Call ``gen_complementary_target()`` if you want to access synthetic complementary labels.

    Parameters
    ----------
    root : str
        path to store dataset file.

    train : bool
        training set if True, else testing set.

    transform : callable, optional
        a function/transform that takes in a PIL image and returns a transformed version.

    target_transform : callable, optional
        a function/transform that takes in the target and transforms it.

    download : bool
        if true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.

    num_cl : int
        the number of real-world complementary labels of each data chosen from [1, 3].

    Attributes
    ----------
    data : Tensor
        the feature of sample set.

    targets : Tensor
        the complementary labels for corresponding sample.

    true_targets : Tensor
        the ground-truth labels for corresponding sample.

    num_classes : int
        the number of classes.

    input_dim : int
        the feature space after data compressed into a 1D dimension.

    """

    label_map = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    
    def __init__(
        self,
        root="./data/cifar100",
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        long_label=False, 
    ):
        super(CIFAR100, self).__init__(
            root, train, transform, target_transform, download
        )
        self.num_classes = 100
        self.input_dim = 3 * 32 * 32
        self.names = [f"{i}" for i in range(50000)]
        self.targets = torch.tensor(self.targets).view(-1, 1)
        if not long_label:
            self.label_map = [label.split(" ")[-1] for label in self.label_map]
        self.class_to_idx = {self.label_map[i]: i for i in range(len(self.label_map))}
    
    @classmethod
    def build_dataset(self, train, long_label, do_transform=False):
        if train:
            if do_transform:
                train_transform = transforms.Compose(
                    [
                        # RandomAugment(3, 5), 
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
                        ),
                    ]
                )
                dataset = self(
                    train=True,
                    transform=train_transform,
                    long_label=long_label, 
                )
            else:
                dataset = self(
                    train=True,
                    long_label=long_label, 
                )
        else:
            if do_transform:
                test_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
                    ]
                )
                dataset = self(
                    train=False,
                    transform=test_transform,
                    long_label=long_label, 
                )
            else:
                dataset = self(
                    train=False,
                    long_label=long_label, 
                )
        return dataset