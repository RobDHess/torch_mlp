from torch_mlp import MLP
import torch.nn as nn

mlp = MLP(d_in=10, d_hidden=20, d_out=1, depth=3, p=0.5, norm=nn.BatchNorm1d)
