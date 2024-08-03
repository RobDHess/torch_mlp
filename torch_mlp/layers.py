import torch
import torch.nn as nn


def get_kwargs(kwargs, key):
    if kwargs is None:
        return {}
    if key in kwargs:
        return kwargs[key]
    return {}


class Creator:
    """Helper class to create layers"""

    def linear(d_in, d_out, bias=True, **kwargs):
        kwargs = get_kwargs(kwargs, "linear_params")
        return nn.Linear(d_in, d_out, bias=bias, **kwargs)

    def norm(norm, d, **kwargs):
        kwargs = get_kwargs(kwargs, "norm_params")
        return norm(d, **kwargs)

    def act(act, **kwargs):
        kwargs = get_kwargs(kwargs, "act_params")
        return act(**kwargs)

    def dropout(dropout, **kwargs):
        kwargs = get_kwargs(kwargs, "dropout_params")
        return nn.Dropout(dropout, **kwargs)

    def embedding(K, d, **kwargs):
        kwargs = get_kwargs(kwargs, "embedding_params")
        return nn.Embedding(K, d, **kwargs)

    def lazy_linear(d_out, bias=True, **kwargs):
        kwargs = get_kwargs(kwargs, "linear_params")
        return nn.LazyLinear(d_out, bias=bias, **kwargs)

    def create_flatten(**kwargs):
        kwargs = get_kwargs(kwargs, "flatten_params")
        return nn.Flatten(**kwargs)


class Initializer:
    def initialise(mlp, weight_method=None, bias_method=None, **kwargs):
        assert (
            weight_method is not None or bias_method is not None
        ), "At least one of weight or bias method must be specified."
        weight_kwargs = get_kwargs(kwargs, "weight_init_params")
        bias_kwargs = get_kwargs(kwargs, "bias_init_params")
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                if weight_method is not None:
                    weight_method(layer.weight, **weight_kwargs)
                if bias_method is not None:
                    bias_method(layer.bias, **bias_kwargs)


class ResidualBlock(nn.Module):
    """Residual block for MLP"""

    def __init__(self, layers, cst_in=None, cst_res=None):
        super(ResidualBlock, self).__init__()
        self.mlp = nn.Sequential(*layers)
        self.cst_in = cst_in
        self.cst_res = cst_res

    def forward(self, x):
        residual = self.mlp(x)
        if self.cst_in is not None:
            x = x * self.cst_in
        if self.cst_res is not None:
            residual = residual * self.cst_res
        return x + residual

    def __repr__(self):
        if self.cst_in is not None or self.cst_res is not None:
            return f"ResidualBlock(cst_in={self.cst_in}, cst_res={self.cst_res}) \n {self.mlp}, "
        return f"ResidualBlock(\n{self.mlp})"
