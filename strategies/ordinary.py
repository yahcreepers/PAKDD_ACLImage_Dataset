import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import math
from libcll.strategies import Strategy

class Ordinary(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        self.log("Train_Loss", loss)
        return loss
