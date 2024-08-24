try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is not installed. Please install PyTorch by following the instructions at https://pytorch.org"
    ) from e


from .mlp import MLP

__all__ = ["MLP"]
