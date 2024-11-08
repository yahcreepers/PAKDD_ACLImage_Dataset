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

class CIFAR10(torchvision.datasets.CIFAR10):
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

    label_map = [
        "airplane", 
        "automobile", 
        "bird", 
        "cat", 
        "deer", 
        "dog", 
        "frog", 
        "horse", 
        "ship", 
        "truck"
    ]
    
    def __init__(
        self,
        root="./data/cifar10",
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        long_label=False, 
    ):
        if train:
            dataset_path = f"{root}/clcifar10.pkl"
            if download and not os.path.exists(dataset_path):
                os.makedirs(root, exist_ok=True)
                gdown.download(
                    id="1uNLqmRUkHzZGiSsCtV2-fHoDbtKPnVt2", output=dataset_path
                )
            with open(dataset_path, "rb") as f:
                data = pickle.load(f)
            self.data = data["images"]
            self.names = data["names"]
            self.targets = torch.Tensor(data["ord_labels"]).view(-1)
            self.transform = transform
            self.target_transform = target_transform
        else:
            super(CIFAR10, self).__init__(
                root, train, transform, target_transform, download
            )
            self.targets = torch.Tensor(self.targets)
        self.num_classes = 10
        self.input_dim = 3 * 32 * 32
        if not long_label:
            self.label_map = [label.split(" ")[-1] for label in self.label_map]
        self.class_to_idx = {self.label_map[i]: i for i in range(len(self.label_map))}
    
    @classmethod
    def build_dataset(self, train, long_label, do_transform=False):
        if train:
            if do_transform:
                train_transform = transforms.Compose(
                    [
                        RandomAugment(3, 5), 
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
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
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
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