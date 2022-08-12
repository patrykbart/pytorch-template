import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        raise NotImplementedError()

    def forward(self, inputs):
        raise NotImplementedError()
