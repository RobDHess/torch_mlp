import torch
import torch.nn as nn
from copy import deepcopy

from .creator import Creator


class MLP(nn.Module):
    def __init__(
        self,
        # Shape
        d_in=None,
        d_hidden=None,
        d_out=None,
        depth=None,
        dims=None,
        # Layer types
        linear=nn.Linear,
        norm=None,
        act=nn.ReLU,
        dropout=nn.Dropout,
        layer=[
            {"module": "linear"},
            {"module": "norm"},
            {"module": "act"},
            {"module": "dropout"},
        ],
        # Common kwargs
        residual=False,
        bias=True,
        p=0,
        # Pre and Post nets
        prelude=None,
        finale=None,
        end_linear=True,
        end_linear_config={"module": "linear"},
        # Initialisation
        weight_init=None,
        bias_init=None,
        embed_init=None,
    ):
        super(MLP, self).__init__()
        # Dimensions
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.depth = depth
        self.dims = dims

        # Layer template
        self.module_dict = {
            "linear": linear,
            "norm": norm,
            "act": act,
            "dropout": dropout,
        }
        self.layer = layer

        # Common configs
        self.residual = residual
        self.bias = bias
        self.p = p

        # Pre and post nets
        self.end_linear = end_linear
        self.end_linear_config = end_linear_config
        self.prelude = prelude
        self.finale = finale

        # Initialisation
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.embed_init = embed_init

        # Process input arguments
        self._check_args()
        self._impute_defaults()
        self._check_layer()
        self._parse_config_mode()

        template = self._create_template()

        # Create layers
        layers = self._create_layers(template)

        self.net = nn.Sequential(*layers)

    def _check_args(self):
        """Check input arguments"""
        # Check shape arguments
        modes = [
            self.dims is not None,
            (
                self.d_hidden is not None
                and self.d_out is not None
                and self.depth is not None
            ),
        ]
        assert (
            sum(modes) == 1
        ), "Must provide either a list of dims or d_in, d_hidden, d_out, and depth"

        # Check special finale argument
        if self.end_linear and self.finale is not None:
            raise ValueError("Cannot have both end_linear and finale")

        try:
            self.p = float(self.p)
        except ValueError:
            raise ValueError("p must be a float")
        assert 0 <= self.p <= 1, "p must be between 0 and 1"

        # # Check residual argument with layer
        # add_in_layer_config = any([l["type"] == "add" for l in self.layer])
        # if self.residual and add_in_layer_config:
        #     raise ValueError("Cannot have both residual=True and add in layer")

    def _check_layer(self):
        """Check the layer configuration"""
        # Remove None layers
        self.layer = [l for l in self.layer if l["module"] is not None]

        linears = [l for l in self.layer if l["type"] == "linear"]
        if len(linears) > 1:
            for l in linears[:-1]:
                if "d_out" not in l:
                    raise ValueError("All but the last linear layer must specify d_out")

    def _parse_config_mode(self):
        """Parse the configuration mode for rectangular MLPs"""
        if self.dims is None:
            self.dims = [self.d_in] + [self.d_hidden] * (self.depth - 1) + [self.d_out]

    def _impute_defaults(self):
        """Impute default values for layer configuration"""
        layer = self.layer
        for conf in layer:
            module = conf["module"]
            if isinstance(module, str):
                assert module in self.module_dict, f"Invalid module string: {module}"
                conf["module"] = self.module_dict[module]
                conf["type"] = module

            if "type" not in conf:
                continue
            conf_type = conf["type"]
            if conf_type == "linear":
                if "bias" not in conf:
                    conf["bias"] = self.bias
            elif conf_type == "dropout":
                if "p" not in conf:
                    if self.p == 0:
                        layer.remove(conf)
                    else:
                        conf["p"] = self.p
        self.layer = layer

        if isinstance(self.end_linear_config["module"], str):
            self.end_linear_config["module"] = self.module_dict[
                self.end_linear_config["module"]
            ]
            self.end_linear_config["type"] = "linear"
        if "bias" not in self.end_linear_config:
            self.end_linear_config["bias"] = self.bias

    def _parse_d_out(self, d, d_in, d_out):
        if isinstance(d, int):
            return d
        if isinstance(d, str):
            if d_in in d:
                remainder = d.replace(d_in, "")
                base = d_in
            elif d_out in d:
                remainder = d.replace(d_out, "")
                base = d_out
            try:
                factor = float(remainder)
                d = round(factor * base)
                return d
            except ValueError:
                raise ValueError(f"Invalid string argument for d_out: {d}")

    def _create_template(self):
        template = []

        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            current_dim = d_in
            layer = []
            is_last = i == len(self.dims) - 2
            for l in deepcopy(self.layer):
                if is_last and self.end_linear:
                    l = self.end_linear_config
                    l["d_in"] = current_dim
                    l["d_out"] = d_out
                    layer.append(l)
                    continue
                if l["type"] == "linear":
                    l["d_in"] = current_dim
                    if "d_out" not in l:
                        l["d_out"] = d_out
                    else:
                        l["d_out"] = self._parse_d_out(l["d_out"], d_in, d_out)
                    current_dim = l["d_out"]

                if l["type"] == "norm":
                    l["d_in"] = current_dim

                layer.append(l)

            template.append(layer)
        return template

    def _create_ends(self, layer):
        """Create prelude or finale layers"""
        if layer is None:
            return []
        if isinstance(layer, MLP):
            return self.prelude
        else:
            return self.create_layer(layer)

    def _create_layers(self, template):
        layers = []
        for layer in template:
            for l in layer:
                layers.append(Creator.create(l))
        return layers
