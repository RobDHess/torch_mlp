import torch
import torch.nn as nn

from .layers import ResidualBlock, Creator, Initializer


class MLP(nn.Module):
    def __init__(
        self,
        dims=None,
        d_in=None,
        d_hidden=None,
        d_out=None,
        depth=None,
        norm=None,
        act=nn.ReLU,
        dropout=None,
        bias=True,
        residual=False,
        flatten=False,
        K=None,
        start_lin=False,
        end_act=None,
        weight_init=None,
        bias_init=None,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.dims = dims
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.depth = depth
        self.norm = norm
        self.act = act
        self.dropout = dropout
        self.bias = bias
        self.residual = residual
        self.flatten = flatten
        self.K = K
        self.embed = K is not None
        self.start_lin = start_lin
        self.end_act = end_act
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.kwargs = kwargs

        self.sanity_check()

        self.config_mode = self._check_config_mode()
        if self.config_mode == "dims":
            self.layers = self.create_from_dims()
        elif self.config_mode == "rect":
            self.layers = self.create_rectangular()

        pre_layers = []
        if self.flatten:
            flatten = Creator.create_flatten(**self.kwargs)
            pre_layers.append(flatten)
        if self.embed:
            if self.config_mode == "dims":
                d_in = dims[0]
            elif self.config_mode == "rect":
                d_in = self.d_in
            assert d_in is not None, "Input dimension must be specified."
            embed = Creator.embedding(K, d_in, **self.kwargs)
            pre_layers.append(embed)

        self.layers = pre_layers + self.layers
        self.mlp = nn.Sequential(*self.layers)

        if self.weight_init is not None or self.bias_init is not None:
            Initializer.initialise(
                self.mlp, self.weight_init, self.bias_init, **self.kwargs
            )

    def forward(self, x):
        return self.mlp(x)

    def sanity_check(self):
        if self.start_lin:
            assert (
                self.residual
            ), "Using start_lin does not make sense without residual connections."

    def _check_config_mode(self):
        modes = [
            self.dims is not None,
            (
                self.d_hidden is not None
                and self.d_out is not None
                and self.depth is not None
            ),
        ]

        if sum(modes) != 1:
            raise ValueError("Only one configuration mode can be used at a time.")

        if self.dims is not None:
            return "dims"
        elif (
            self.d_in is not None
            and self.d_hidden is not None
            and self.d_out is not None
            and self.depth is not None
        ):
            return "rect"
        else:
            raise ValueError("Invalid configuration.")

    def _create_layer(self, d_in, d_out, lin_only=False):
        """Create a single layer [linear, norm, act, dropout]"""
        layers = []
        if d_in is None:
            linear = Creator.lazy_linear(d_out, bias=self.bias, **self.kwargs)
        else:
            linear = Creator.linear(d_in, d_out, bias=self.bias, **self.kwargs)
        layers.append(linear)

        if lin_only:
            return layers

        if self.norm is not None:
            norm = Creator.norm(self.norm, d_out, **self.kwargs)
            layers.append(norm)
        if self.act is not None:
            act = Creator.act(self.act, **self.kwargs)
            layers.append(act)
        if self.dropout is not None:
            dropout = Creator.dropout(self.dropout, **self.kwargs)
            layers.append(dropout)
        return layers

    def create_from_dims(self):
        for x in self.dims:
            assert isinstance(x, int) or isinstance(
                x, None
            ), "All elements in dims must be integers or None."

        layers = []
        dims = self.dims
        for i in range(len(dims) - 2):
            d_in = dims[i]
            d_out = dims[i + 1]
            layer = self._create_layer(d_in, d_out, lin_only=self.start_lin and i == 0)
            if self.residual and d_in == d_out and i != 0:
                layer = [ResidualBlock(layer)]
            layers += layer

        layers += self.create_tail(dims[-2], dims[-1])
        return layers

    def create_rectangular(self):
        assert self.depth >= 1, "Depth must be at least 1."

        layers = []
        layers += self._create_layer(self.d_in, self.d_hidden, lin_only=self.start_lin)
        for i in range(self.depth - 1):
            layer = self._create_layer(self.d_hidden, self.d_hidden)
            if self.residual:
                layer = [ResidualBlock(layer)]
            layers += layer

        layers += self.create_tail(self.d_hidden, self.d_out)
        return layers

    def create_tail(self, d_in, d_out):
        layers = []
        end_lin = Creator.linear(d_in, d_out, bias=self.bias, **self.kwargs)
        layers.append(end_lin)
        if self.end_act is not None:
            if self.end_act == self.act:
                end_act = Creator.act(self.end_act, **self.kwargs)
            else:
                end_act = Creator.act(self.end_act, **{})
            layers.append(end_act)
        return layers

    def __repr__(self):
        return f"MLP({self.mlp})"
