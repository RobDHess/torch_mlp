# <img src="torch_mlp.png" alt="Logo" width="150" align="left"/> Torch MLP 

`torch-mlp` is a Python package built on PyTorch for creating flexible and customizable multi-layer perceptrons (MLPs). This package provides an easy way to construct MLPs with various configurations, including custom activation functions, normalization layers, and dropout.

## Features

- Easily configurable MLP architectures
- Supports custom layers like batch normalization, dropout, and various activation functions
- Lazy initialization and embedding support
- Weight and bias initialization options

## Installation

Before installing `torch-mlp`, ensure that you have PyTorch installed. You can install PyTorch by following the instructions at the [official PyTorch website](https://pytorch.org/get-started/locally/).

Once PyTorch is installed, you can install `torch-mlp` via pip:

```bash
pip install torch-mlp
```

## Basic Usage

Each MLP layer is a sequence of `linear`->`normalisation`->`activation`->`dropout`. Here’s a quick example of how to use the `torch-mlp` package:

```python
import torch
import torch.nn as nn
from torch_mlp import MLP

# Define the MLP architecture
mlp = MLP(
    d_in=10,          # Input dimension
    d_hidden=20,      # Hidden layer dimension
    d_out=5,          # Output dimension
    depth=2,          # Number of hidden layers
    act=nn.ReLU,  # Activation function
    norm=nn.BatchNorm1d,  # Normalization layer
    p=0.1,            # Dropout probability
)

# Create a random input tensor
x = torch.randn(4, 10)  # Batch size 4, input dimension 10

# Perform a forward pass
output = mlp(x)
```
If you print this MLP, you will see the following architecture:

```python
MLP(
  (net): Sequential(
    (0): Linear(in_features=10, out_features=20, bias=True)
    (1): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.1, inplace=False)
    (4): Linear(in_features=20, out_features=20, bias=True)
    (5): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Dropout(p=0.1, inplace=False)
    (8): Linear(in_features=20, out_features=5, bias=True)
  )
)
```

## Class `MLP`

### Initialization

The `MLP` class is initialized with the following parameters:

```python
MLP(
    d_in: int = None,
    d_hidden: int = None,
    d_out: int = None,
    depth: int = None,
    dims: list[Optional[int]] = None,
    linear: Type[nn.Module] = nn.Linear,
    norm: Optional[Type[nn.Module]] = None,
    act: Type[nn.Module] = nn.ReLU,
    dropout: Optional[Type[nn.Module]] = nn.Dropout,
    bias: bool = True,
    p: Optional[float] = None,
    embed: bool = False,
    end_linear: bool = True,
    end_act: Optional[Type[nn.Module]] = None,
    linear_kwargs: Optional[dict] = None,
    norm_kwargs: Optional[dict] = None,
    act_kwargs: Optional[dict] = None,
    dropout_kwargs: Optional[dict] = None,
    weight_init: Optional[callable] = None,
    bias_init: Optional[callable] = None,
    weight_kwargs: Optional[dict] = None,
    bias_kwargs: Optional[dict] = None,
)
```

### Parameters

##### Shape
- **`d_in`**: The dimension of the input layer.
- **`d_hidden`**: The dimension of each hidden layer.
- **`d_out`**: The dimension of the output layer.
- **`depth`**: The number of layers.
- **`dims`**: Alternatively, you can specify a list of dimensions for each layer.
##### Layer types
- **`linear`**: The linear layer module to use. Defaults to `nn.Linear`.
- **`norm`**: Optional normalization layer module (e.g., `nn.BatchNorm1d`).
- **`act`**: Activation function module. Defaults to `nn.ReLU`.
- **`dropout`**: Optional dropout module. Defaults to `nn.Dropout`.
##### Layer parameters
- **`bias`**: Whether to include bias in the linear layers. Defaults to `True`.
- **`p`**: Dropout probability. Only used if `dropout` is specified.
##### Special structure parameters
- **`embed`**: If `True`, the first layer will be an embedding layer. `d_in` is used as `num_embeddings`.
- **`end_linear`**: Whether to use a linear only for the final layer. Defaults to `True`.
- **`end_act`**: Optional activation function for the output layer.
##### Layer kwargs
- **`linear_kwargs`**: Additional keyword arguments for the linear layers.
- **`norm_kwargs`**: Additional keyword arguments for the normalization layers.
- **`act_kwargs`**: Additional keyword arguments for the activation functions.
- **`dropout_kwargs`**: Additional keyword arguments for the dropout layers.
##### Initialisation
- **`weight_init`**: Custom weight initialization function.
- **`bias_init`**: Custom bias initialization function.
- **`weight_kwargs`**: Additional keyword arguments for weight initialization.
- **`bias_kwargs`**: Additional keyword arguments for bias initialization.

**Note**: kwargs take precedence over arguments. For example, if you specify `bias=False` in the initialization and set `linear_kwargs={"bias":True}`, the linear layers will have bias.

### Example Usages

1. **Basic MLP with ReLU Activation:**
   ```python
   mlp = MLP(d_in=10, d_hidden=20, d_out=5, depth=3)
   ```
   or 
    ```python
    mlp = MLP(dims=[10, 20, 20, 5])
    ```

2. **MLP with Custom Layers:**
   ```python
   mlp = MLP(
       d_in=10,
       d_hidden=20,
       d_out=5,
       depth=3,
       act=nn.GELU,
       norm=nn.LayerNorm,
   )
   ```

3. **MLP with Lazy Initialization:**
   ```python
   mlp = MLP(d_in=None, d_hidden=20, d_out=5, depth=3)
   ```
   or 
    ```python
    mlp = MLP(dims=[None, 20, 20, 5])
    ```

4. **MLP with Embedding Layer:**
   ```python
   mlp = MLP(d_in=10, d_hidden=20, d_out=5, depth=3, embed=True)
   ```

5. **MLP with Custom Initialization:**
   ```python
    mlp = MLP(
         d_in=10,
         d_hidden=20,
         d_out=5,
         depth=3,
         weight_init=nn.init.kaiming_normal_,
         bias_init=nn.init.zeros_,
         weight_kwargs={"mode": "fan_in", "nonlinearity": "relu"},
    )
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any questions, feel free to reach out via [GitHub Issues](https://github.com/robdhess/torch-mlp/issues).

