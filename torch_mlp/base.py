import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    def forward(self, x):
        if self.flatten is not None:
            pass
