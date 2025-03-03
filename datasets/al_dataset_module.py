import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, Sampler, DataLoader
import numpy as np
import os
from .utils import prepare_dataset
from libcll.datasets.utils import collate_fn_multi_label, collate_fn_one_hot
from .cifar20 import _cifar100_to_cifar20

class IndexSampler(Sampler):
    def __init__(self, index):
        self.index = index

    def __iter__(self):
        ind = torch.randperm(len(self.index))
        return iter(self.index[ind].tolist())

    def __len__(self):
        return len(self.index)

class ALDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        args, 
    ):
        super().__init__()
        self.args = args
        self.args.one_hot = True if self.args.strategy == "MCL" else False
    
    def setup(self, stage=None):
        pl.seed_everything(self.args.seed, workers=True)
        self.train_set, self.test_set = prepare_dataset(self.args)
        self.train_set.true_targets = self.train_set.targets.clone()
        if self.args.dataset.startswith("cl"):
            self.train_set.targets = self.train_set.cls
        if self.args.dataset == "cifar20":
            with open(os.path.join(self.args.label_path, "auto_labels.csv"), "r") as f:
                labels = [[self.train_set.class_to_idx[l] for l in line.strip().split(",", 1)[1:]] for line in f.readlines()]
            self.train_set.targets = torch.tensor(labels).view(-1, 1)
        else:
            with open(os.path.join(self.args.label_path, "auto_labels.csv"), "r") as f:
                labels = [[self.train_set.class_to_idx[l] for l in line.strip().split(",")[1:]] for line in f.readlines()]
            self.train_set.targets = torch.tensor(labels)[:, :self.args.num_cl]
        
        idx = np.arange(len(self.train_set))
        np.random.shuffle(idx)
        self.train_idx = idx[: int(len(self.train_set) * (1 - self.args.valid_split))]
        self.valid_idx = idx[int(len(self.train_set) * (1 - self.args.valid_split)) :]
        self.train_set.targets = [i for i in self.train_set.targets]
        if self.args.valid_type == "Accuracy":
            for i in self.valid_idx:
                self.train_set.targets[i] = self.train_set.true_targets[i].view(1)

    def train_dataloader(self):
        train_sampler = IndexSampler(self.train_idx)
        train_loader = DataLoader(
            self.train_set,
            sampler=train_sampler,
            batch_size=self.args.batch_size,
            collate_fn=(
                collate_fn_multi_label
                if not self.args.one_hot
                else lambda batch: collate_fn_one_hot(
                    batch, num_classes=self.train_set.num_classes
                )
            ),
            shuffle=False,
            num_workers=4, 
        )
        return train_loader

    def val_dataloader(self):
        if self.args.valid_split:
            valid_sampler = IndexSampler(self.valid_idx)
            valid_loader = DataLoader(
                self.train_set,
                sampler=valid_sampler,
                batch_size=self.args.batch_size,
                collate_fn=collate_fn_multi_label,
                shuffle=False,
                num_workers=4, 
            )
        else:
            valid_loader = DataLoader(
                self.test_set, 
                batch_size=self.args.batch_size, 
                shuffle=False, 
                num_workers=8, 
                persistent_workers=True, 
                pin_memory=True, 
            )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=4, 
        )
        return test_loader
    
    def get_distribution_info(self):
        Q = torch.zeros((self.train_set.num_classes, self.train_set.num_classes))
        for idx in self.train_idx:
            Q[self.train_set.true_targets[idx].long()] += torch.histc(
                self.train_set.targets[idx].float(), self.train_set.num_classes, 0, self.train_set.num_classes
            )
        class_priors = Q.sum(dim=0)
        Q = Q / Q.sum(dim=1).view(-1, 1)
        return (
            Q,
            class_priors,
        )
