import torch.nn as nn


class Creator:
    """Helper class to create layers"""

    def create(config):
        """Create layer from config dictionary."""
        print(config)
        module_type = config.pop("type")
        module = config.pop("module")
        if module_type == "linear":
            d_in = config.pop("d_in")
            d_out = config.pop("d_out")
            return module(d_in, d_out, **config)
        if module_type == "norm":
            d_in = config.pop("d_in")
            return module(d_in, **config)
        else:
            return module(**config)
